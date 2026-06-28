"""
Prometheus HTTP instrumentation middleware.
Tracks request count and latency for every API path.
Strips numeric path segments so /api/query/123 → /api/query/{id}
to avoid high-cardinality label explosion.
"""
from __future__ import annotations

import re
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from metrics.prometheus_metrics import (
    http_requests_total,
    http_request_duration_seconds,
    errors_total,
)

_NUMERIC_RE = re.compile(r"/\d+")


def _normalize_path(path: str) -> str:
    """Replace numeric path segments with {id} to limit cardinality."""
    return _NUMERIC_RE.sub("/{id}", path)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        path = _normalize_path(request.url.path)
        method = request.method
        start = time.perf_counter()

        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception as exc:
            errors_total.labels(endpoint=path, error_type=type(exc).__name__).inc()
            raise

        elapsed = time.perf_counter() - start
        http_requests_total.labels(method=method, path=path, status=status).inc()
        http_request_duration_seconds.labels(method=method, path=path).observe(elapsed)
        return response
