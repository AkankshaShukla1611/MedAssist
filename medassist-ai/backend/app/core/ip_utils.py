"""
Single source of truth for "what is the client's IP address", used by both
the rate limiter (app.core.limiter) and the audit log
(app.services.audit_service). Previously these each had their own
implementation and disagreed — the rate limiter used raw
`request.client.host` while the audit log correctly parsed
`X-Forwarded-For`, meaning rate limiting was effectively per-proxy (i.e.
global, not per-client) in the documented nginx-fronted deployment.

Trust note: X-Forwarded-For is only trustworthy when set by a proxy YOU
control (docker/nginx.conf does set it correctly here). If this app is ever
deployed without a trusted proxy in front of it, an attacker could spoof
this header to evade rate limiting — in that topology, set
TRUST_PROXY_HEADERS=False (or simply don't route through nginx without
also stripping/overwriting X-Forwarded-For at the edge).
"""
from fastapi import Request


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # X-Forwarded-For can be a comma-separated chain (client, proxy1, proxy2, ...)
        # — the first entry is the original client.
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
