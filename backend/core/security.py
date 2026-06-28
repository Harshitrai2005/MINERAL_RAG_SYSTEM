"""
API Security Module
─────────────────────────────────────────────────────────────────────────────
Handles:
  - API Key validation (X-API-Key header)
  - Rate limiting (requests per minute per IP)
  - GROQ_API_KEY startup validation
  - Request authentication for protected endpoints
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Optional
from functools import wraps

from fastapi import Depends, HTTPException, Request, Header
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ── API Key Validation ──────────────────────────────────────────────────────

def validate_api_key(api_key: str = Header(None, alias="X-API-Key")) -> str:
    """
    Dependency for protecting API endpoints with API key authentication.
    
    Usage:
        @router.post("/upload")
        async def upload(file: UploadFile, authenticated: str = Depends(validate_api_key)):
            # authenticated parameter will contain the API key if valid
    
    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    from core.config import settings
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include 'X-API-Key' header.",
        )
    
    if api_key != settings.API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key[:5]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )
    
    return api_key


def validate_groq_api_key(groq_key: Optional[str]) -> bool:
    """
    Validate GROQ_API_KEY at startup.
    
    Returns:
        bool: True if valid, False if missing/invalid
    
    Raises:
        RuntimeError: If GROQ_API_KEY is not set
    """
    if not groq_key or groq_key == "placeholder":
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and set it in .env"
        )
    
    if len(groq_key) < 10:
        raise RuntimeError(
            "GROQ_API_KEY appears invalid (too short). "
            "Get a valid key at https://console.groq.com"
        )
    
    logger.info("[OK] GROQ_API_KEY validated at startup")
    return True


# ── Rate Limiting ──────────────────────────────────────────────────────────

class RateLimiter:
    """
    In-memory rate limiter tracking requests per IP address.
    
    Production note: For distributed systems, use Redis or similar.
    This implementation is suitable for single-instance deployments.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: defaultdict[str, list[float]] = defaultdict(list)
    
    def is_rate_limited(self, client_ip: str) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_ip: IP address of the client
            
        Returns:
            bool: True if rate limited, False if allowed
        """
        if self.requests_per_minute == 0:
            return False  # Rate limiting disabled
        
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests (older than 1 minute)
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check if over limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return True
        
        # Record this request
        self.requests[client_ip].append(now)
        return False
    
    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for this minute."""
        if self.requests_per_minute == 0:
            return -1  # Unlimited
        
        now = time.time()
        minute_ago = now - 60
        recent = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        return max(0, self.requests_per_minute - len(recent))


# Global rate limiters for different endpoints
query_rate_limiter = RateLimiter(requests_per_minute=60)
ingest_rate_limiter = RateLimiter(requests_per_minute=30)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting specific endpoints.
    """
    
    def __init__(self, app, rate_limiter: RateLimiter, paths: list[str]):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.paths = paths
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP (handle proxies: X-Forwarded-For, X-Real-IP)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.headers.get("X-Real-IP", request.client.host)
        
        # Check if this request path should be rate limited
        should_limit = any(request.url.path.startswith(path) for path in self.paths)
        
        if should_limit:
            if self.rate_limiter.is_rate_limited(client_ip):
                remaining = self.rate_limiter.get_remaining(client_ip)
                logger.warning(f"Rate limit exceeded for IP {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded. Max 60 requests per minute.",
                        "retry_after": 60,
                    },
                    headers={"Retry-After": "60"},
                )
            
            response = await call_next(request)
            remaining = self.rate_limiter.get_remaining(client_ip)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests_per_minute)
            return response
        
        return await call_next(request)


# ── MIME Type Validation ────────────────────────────────────────────────────

ALLOWED_MIME_TYPES = {
    # Reports (PDF/TXT)
    "application/pdf": [".pdf"],
    "text/plain": [".txt"],
    "application/x-pdf": [".pdf"],
    
    # Datasets (CSV/JSON)
    "text/csv": [".csv"],
    "application/csv": [".csv"],
    "application/json": [".json"],
    "text/json": [".json"],
}

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".json"}


def validate_file_mime_type(filename: str, content_type: Optional[str], content: bytes) -> tuple[bool, str]:
    """
    Validate uploaded file's MIME type and extension.
    
    Args:
        filename: Original filename from upload
        content_type: MIME type from Content-Type header
        content: File content bytes
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check extension
    import os
    _, ext = os.path.splitext(filename.lower())
    
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type '{ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check MIME type if provided
    if content_type:
        base_mime = content_type.split(";")[0].strip()
        if base_mime not in ALLOWED_MIME_TYPES:
            # Warn but don't reject—some clients send wrong MIME types
            logger.warning(f"Unexpected MIME type {base_mime} for {filename}")
    
    # Magic number validation (file signatures)
    if ext == ".pdf" and not content.startswith(b"%PDF"):
        return False, "File does not appear to be a valid PDF (invalid signature)"
    
    if ext == ".json":
        try:
            import json
            json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False, "File does not appear to be valid JSON"
    
    if ext == ".csv":
        # CSV is flexible, just check it's readable text
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            return False, "File does not appear to be valid UTF-8 text"
    
    return True, ""


# ── Domain Validation (v6) ────────────────────────────────────────────────────
# Only accept files with geo/mineral/geochemistry content.
# Legal, financial, medical, and other non-domain files are rejected.

_GEO_KEYWORDS = {
    # Geological
    "geology", "geological", "geologic", "lithology", "stratigraphy", "petrology",
    "mineralogy", "mineralization", "alteration", "intrusion", "extrusion",
    "plutonic", "volcanic", "metamorphic", "sedimentary", "igneous",
    # Mineral exploration
    "mineral", "minerals", "exploration", "prospect", "deposit", "ore", "assay",
    "drilling", "drillhole", "borehole", "intercept", "grade", "zone",
    "ppm", "g/t", "ppb", "geochemical", "geochemistry",
    # Elements / commodities
    "gold", "silver", "copper", "zinc", "lead", "molybdenum", "tungsten",
    "au", "ag", "cu", "zn", "pb", "mo", "w", "fe", "mn", "as",
    "lithium", "cobalt", "nickel", "uranium", "platinum",
    # Survey / mapping
    "survey", "mapping", "sampling", "sample", "core", "chip", "rock", "soil",
    "anomaly", "anomalous", "pathfinder", "porphyry", "epithermal", "vms",
    "skarn", "iocg", "sedex", "orogenic",
    # Data fields
    "easting", "northing", "elevation", "azimuth", "dip", "depth",
    "formation", "member", "unit", "section",
}

_REJECTED_KEYWORDS = {
    "legal", "contract", "agreement", "invoice", "receipt", "warranty",
    "terms of service", "privacy policy", "court", "lawsuit", "attorney",
    "medical", "diagnosis", "treatment", "patient", "prescription",
    "financial statement", "balance sheet", "income statement", "tax return",
}


def validate_domain_content(filename: str, content: bytes, category: str) -> tuple[bool, str]:
    """
    Validate that uploaded file content belongs to the mineral/geo domain.
    Rejects legal documents, medical records, financial statements, etc.

    Returns (is_valid, message)
    """
    import os
    _, ext = os.path.splitext(filename.lower())

    # For PDFs: check filename heuristics (full text extraction too heavy here)
    sample_text = ""
    if ext == ".pdf":
        # Check filename for obvious non-domain signals
        fname_lower = filename.lower()
        for bad in _REJECTED_KEYWORDS:
            if bad.replace(" ", "_") in fname_lower or bad.replace(" ", "-") in fname_lower:
                return False, (
                    f"File '{filename}' appears to be a {bad.split()[0]} document, "
                    "not a mineral exploration report. "
                    "Only geological reports, assay data, and geochemical datasets are accepted."
                )
        # Check first 2KB of text for domain keywords
        try:
            sample_text = content[:2048].decode("utf-8", errors="ignore").lower()
        except Exception:
            pass

    elif ext in (".csv", ".json"):
        # Check first 1KB for domain keywords
        try:
            sample_text = content[:1024].decode("utf-8", errors="ignore").lower()
        except Exception:
            pass

    if sample_text:
        # Check for rejected content
        for bad in _REJECTED_KEYWORDS:
            if bad in sample_text:
                # Only reject if NO geo keywords present at all
                has_geo = any(kw in sample_text for kw in _GEO_KEYWORDS)
                if not has_geo:
                    return False, (
                        f"File '{filename}' appears to contain non-geological content ({bad}). "
                        "Only mineral exploration, geochemistry, and geological data are accepted."
                    )

    return True, ""
