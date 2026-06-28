#!/usr/bin/env python3
"""
Seed Sample Data Script
─────────────────────────────────────────────────────────────────────────────
Ingests all files from sample_data/ into the running MEIS backend.
Run this after the app is started to populate the knowledge base for testing.

Usage:
    python scripts/seed_sample_data.py [--base-url http://localhost:8000] [--api-key YOUR_KEY]

Requirements:
    pip install httpx python-dotenv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY  = os.getenv("API_KEY", "test-key")

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"

# Maps file extension → (endpoint, category)
# The real upload endpoint is /api/ingest/upload for all file types.
# Category tells the ingestion service which processor to use:
#   "report"  → PDFProcessor / text chunker  (for .pdf, .txt)
#   "dataset" → MineralDatasetProcessor      (for .csv, .json)
INGEST_MAP = {
    ".txt":  ("/api/ingest/upload", "report"),
    ".pdf":  ("/api/ingest/upload", "report"),
    ".csv":  ("/api/ingest/upload", "dataset"),
    ".json": ("/api/ingest/upload", "dataset"),
}

# Sub-directories (and files) inside sample_data/ to skip
SKIP_DIRS = {"images", "pdfs"}   # pdfs/ contains .txt fakes — keep if you want them


def wait_for_health(client: httpx.Client, max_wait: int = 120) -> bool:
    print(f"⏳ Waiting for backend at {client.base_url}...")
    for i in range(max_wait // 5):
        try:
            r = client.get("/api/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                print(f"✅ Backend healthy — vector_backend={data.get('vector_backend')}")
                return True
        except Exception:
            pass
        print(f"   [{i*5}s] not ready yet...")
        time.sleep(5)
    return False


def ingest_file(client: httpx.Client, file_path: Path, endpoint: str, category: str) -> bool:
    suffix = file_path.suffix.lower()
    content_type = {
        ".txt":  "text/plain",
        ".pdf":  "application/pdf",
        ".csv":  "text/csv",
        ".json": "application/json",
    }.get(suffix, "application/octet-stream")

    print(f"  📄 Ingesting {file_path.name} → {endpoint}  (category={category})")
    try:
        with open(file_path, "rb") as f:
            r = client.post(
                endpoint,
                files={"file": (file_path.name, f, content_type)},
                data={"category": category},
                timeout=120,
            )
        if r.status_code in (200, 201, 202):
            data = r.json()
            print(f"     ✅ {data.get('message', 'ok')} — chunks={data.get('chunks_added', '?')}")
            return True
        else:
            print(f"     ❌ HTTP {r.status_code}: {r.text[:300]}")
            return False
    except Exception as exc:
        print(f"     ❌ Error: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Seed MEIS with sample data")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key",  default=DEFAULT_API_KEY)
    parser.add_argument("--skip-wait", action="store_true", help="Skip health-check wait")
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key}
    client  = httpx.Client(base_url=args.base_url, headers=headers)

    if not args.skip_wait:
        if not wait_for_health(client):
            print("❌ Backend not ready after 2 minutes. Is it running?")
            sys.exit(1)

    if not SAMPLE_DIR.exists():
        print(f"❌ sample_data directory not found at {SAMPLE_DIR}")
        sys.exit(1)

    files = []
    for ext, (endpoint, category) in INGEST_MAP.items():
        for f in SAMPLE_DIR.rglob(f"*{ext}"):
            if f.is_file() and not any(part in SKIP_DIRS for part in f.parts):
                files.append((f, endpoint, category))

    if not files:
        print(f"❌ No sample files found in {SAMPLE_DIR}")
        sys.exit(1)

    print(f"\n🚀 Seeding {len(files)} files into {args.base_url}\n")

    successes, failures = 0, 0
    for file_path, endpoint, category in sorted(files):
        ok = ingest_file(client, file_path, endpoint, category)
        if ok:
            successes += 1
        else:
            failures += 1
        time.sleep(0.5)  # Be gentle with the API

    print(f"\n{'='*50}")
    print(f"✅ Succeeded: {successes}")
    print(f"❌ Failed:    {failures}")

    if successes > 0:
        r = client.get("/api/health")
        if r.status_code == 200:
            data = r.json()
            print(f"\n📊 Knowledge base now has {data.get('total_chunks', '?')} chunks")
            for col, count in data.get("collections", {}).items():
                print(f"   {col}: {count} chunks")

    print(f"\n🎉 Done! Open {args.base_url} in your browser to start querying.")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
