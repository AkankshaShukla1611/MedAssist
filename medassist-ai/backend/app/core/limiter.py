from slowapi import Limiter

from app.core.ip_utils import get_client_ip

# Keyed by client IP via get_client_ip, which respects X-Forwarded-For when
# set by a trusted proxy (see docker/nginx.conf) — using slowapi's default
# get_remote_address here would rate-limit by the PROXY's IP for every user
# behind Nginx, effectively making the limit global instead of per-client.
limiter = Limiter(key_func=get_client_ip)
