import json
from .monitoring import get_logger
from .redis_client import redis_client
from langchain_core.messages import AIMessage, HumanMessage

logger = get_logger()

class ConversationStore:
    """Chat history per thread, stored in Redis so it survives restarts and is shared by workers."""

    def __init__(self, max_turns: int = 6,ttl_seconds: int = 3600, prefix: str = "business_rag_chat:history:"):
        self.max_messages = max_turns * 2
        self.ttl = ttl_seconds
        self.prefix = prefix

    def _key(self, thread_id: str) -> str:
        return f"{self.prefix}{thread_id}"

    async def get(self, thread_id: str) -> list:
        try:
            raw = await redis_client.lrange(self._key(thread_id), 0, -1)
        except Exception as e:
            logger.warning(f"History read failed: {e}")
            return []
        messages = []
        for item in raw:
            d = json.loads(item)
            message_type = HumanMessage if d["role"] == "human" else AIMessage
            messages.append(message_type(content=d["content"]))
        return messages

    async def append(self, thread_id: str, question: str, answer: str) -> None:
        key = self._key(thread_id)
        try:
            async with redis_client.pipeline(transaction=True) as pipe:
                pipe.rpush(
                    key,
                    json.dumps({"role": "human", "content": question}),
                    json.dumps({"role": "ai", "content": answer}),
                )
                pipe.ltrim(key, -self.max_messages, -1)
                pipe.expire(key, self.ttl)
                await pipe.execute()
        except Exception as e:
            logger.warning(f"History write failed: {e}")