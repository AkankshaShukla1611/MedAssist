from slowapi import Limiter
from slowapi.util import get_remote_address

# Keyed by client IP. Behind a reverse proxy, make sure X-Forwarded-For
# is trusted/parsed correctly (see Nginx config) or this will rate-limit
# by the proxy's IP for everyone.
limiter = Limiter(key_func=get_remote_address)
