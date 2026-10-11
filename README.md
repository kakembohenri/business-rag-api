# Parts Unlimited Knowledge Assistant

A retrieval-augmented (RAG) chat API that answers questions about Parts Unlimited East Africa using only the company's own documents. It streams progress and the final answer to a browser chat page, remembers the conversation, and filters unsafe input and output.

## How it works

```
Browser ──> nginx ──> FastAPI app ──> Redis (chat history, response cache)
                          │
                          ├──> Pinecone  (finds the 3 most relevant document chunks)
                          └──> OpenAI    (rewrites follow-ups, writes the answer)
```

For each question the app:

1. Checks the message for prompt injection and masks personal data (phone, email, card, SSN).
2. Looks up the conversation history for the `thread_id`.
3. Returns a cached answer if this exact question was asked before (first messages only).
4. Rewrites follow-ups ("and for Kenya?") into standalone questions, so retrieval works.
5. Retrieves relevant chunks from Pinecone and asks the model to answer from them only.
6. Validates the answer (secrets, invented personal data, prompt leaks), then saves it and sends it.

If the documents don't contain the answer, the assistant says so instead of guessing.

## Project structure

```
app/
  main.py          API routes, streaming endpoint, rate limiting
  retriever.py     prompts, retrieval, and the LangChain chain
  security.py      input screening, PII masking, output validation
  history.py       conversation history (Redis)
  redis_client.py  shared Redis connection
  cache.py         response cache
  config.py        settings (read from environment / .env)
  models.py        request and response models
  monitoring.py    logging and request timing
ui/index.html      browser chat page
nginx/default.conf serves the page and proxies /api to the app
Dockerfile
docker-compose.yml
pyproject.toml / uv.lock
```

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/), or Docker
- Redis
- An OpenAI API key
- A Pinecone index that already contains your document chunks, created with **1024-dimension** embeddings (the code requests `dimensions=1024`)

## Configuration

Create a `.env` file in the project root. It holds secrets, so never commit it.

```dotenv
OPENAI_API_KEY=...
PINECONE_API_KEY=...
INDEX_NAME=...
MODEL_NAME=...                  # OpenAI chat model
EMBEDDINGS_MODEL_NAME=...       # must match how the index was built
REDIS_URL=redis://localhost:6379/0
CANARY=CANARY-<random hex>      # same value on every instance
RATE_LIMIT=10/minute            # per client IP
CACHE_TTL_SECONDS=3600
APP_ENV=development
```

Variable names come from the fields in `config.py`, so check them if something is not picked up. Generate a canary with:

```bash
python -c "import secrets; print('CANARY-' + secrets.token_hex(8))"
```

## Run locally

```bash
uv sync
docker run -d -p 6379:6379 redis:7-alpine      # if you have no Redis
uvicorn app.main:app --reload --port 8000      # run from the project root
curl http://localhost:8000/health
```

## Run with Docker

```bash
docker compose up --build -d
docker compose ps            # services should be running; the API "healthy"
docker compose logs -f
```

Open `http://localhost` for the chat page. Only nginx is exposed. The API and Redis are reachable inside the Docker network only. In `docker-compose.yml`, set `REDIS_URL=redis://redis:6379/0` (inside Docker, `localhost` means the container itself).

## API

**`POST /chat`** streams server-sent events.

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the warranty on brake pads?", "thread_id": "my-first-thread"}'
```

- `thread_id` identifies one conversation. Use a unique, unguessable id per conversation (a UUID), and send the same id on every message. A new id starts a fresh conversation.
- Events: `status` (progress), `sources`, `answer` (final text), `done`, `error`.
- Errors before streaming starts come back as normal HTTP codes: `422` invalid input, `429` rate limit exceeded.

**`GET /health`** returns the status of the agent, security layer, and cache.

## Security

- Blocks common prompt-injection phrasing and caps input at 4000 characters
- Masks personal data before it reaches the model, logs, or cache
- Redacts secret-looking strings and unexpected personal data in answers
- A hidden canary string in the system prompt detects prompt leaks
- Per-IP rate limiting; the API port is never published directly

These filters are a first line of defense, not a guarantee. Keep the index free of sensitive documents and review who can write to it.

## Production notes

- Serve over HTTPS (browsers also require it for some features, like `crypto.randomUUID`).
- Keep port 8000 unpublished, or bound to `127.0.0.1` if you use nginx on the host.
- If you run several workers, point the rate limiter at Redis so limits are shared.
- Tune `score_threshold` in `retriever.py` with real questions.

## Troubleshooting

- **`No module named 'redis.asyncio'`**: a local file named `redis.py` is shadowing the library. Use `redis_client.py`.
- **Container shows unhealthy**: the image has no `curl`, so the healthcheck must use Python.
- **`host not found in upstream`**: the service name in `nginx/default.conf` must match the compose service name.
- **Follow-up questions are not understood**: the `thread_id` changes between messages, or Redis is unreachable. Check `docker compose logs`.
- **Every user shares one rate limit**: uvicorn must trust the proxy headers, and nginx must set `X-Forwarded-For`.
