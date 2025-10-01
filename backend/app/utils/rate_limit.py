# backend/app/utils/rate_limit.py
import time
import redis
from backend.app.config.settings import settings

r = redis.from_url(settings.REDIS_URL)

def allow(host: str, limit_per_min: int = None) -> bool:
    limit = limit_per_min or settings.RATE_LIMIT_PER_MIN
    key = f"ratelimit:{host}:{int(time.time() // 60)}"
    count = r.incr(key)
    if count == 1:
        r.expire(key, 70)
    return count <= limit
