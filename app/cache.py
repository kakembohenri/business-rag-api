import hashlib
from .redis_client import redis_client

class ResponseCache:

    def __init__(self, ttl_seconds=300):
        self.redis = redis_client
        self.ttl = ttl_seconds

    def _make_key(self, query: str):
        normalized = query.lower().strip()

        return (
            "chat:"
            + hashlib.sha256(
                normalized.encode()
            ).hexdigest()
        )

    async def get(self, query: str):
        key = self._make_key(query)

        return await self.redis.get(key)

    async def set(self, query: str, response: str):
        key = self._make_key(query)

        await self.redis.set(
            key,
            response,
            ex=self.ttl,
        )