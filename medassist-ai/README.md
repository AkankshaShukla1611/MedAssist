# MedAssist AI — Clinical RAG Backend

Enterprise-track AI-powered Clinical Decision Support backend: hybrid
retrieval (dense + BM25 + RRF fusion), medical query expansion, calibrated
confidence, hallucination detection, rich citations, session-aware
conversation memory, a fully API-independent RAG evaluation framework,
Celery-backed async processing, Redis caching, audit logging, file
integrity verification, and versioned APIs. Frontend not yet started —
this is backend-first by design, stabilized before any UI work begins.

## Current status (last stabilization pass)

- **158 tests passing, 76% line coverage** (verified by running the suite,
  not estimated — see `backend/tests/`)
- **API versioning**: every route served at both `/api/v1/...` (canonical)
  and its original unversioned path (deprecated, marked via a `Deprecation`
  response header, fully backward compatible)
- **Async processing**: PDF ingestion, embedding, FAISS/BM25 indexing, and
  evaluation runs all execute via Celery workers, not blocking API requests
- **Migrations**: real Alembic migrations exist and are verified to apply
  and roll back cleanly (`backend/alembic/`) — no longer relying solely on
  `create_all()` in production
- **Audit logging**: authentication events, uploads, deletes, and
  evaluation runs are all recorded with user/IP/resource/outcome, queryable
  via `GET /admin/audit-logs`
- **File integrity**: every upload is SHA-256 checksummed at upload time,
  re-verified before ingestion, and duplicate uploads are detected and
  rejected

## What's implemented

- **Auth**: register / login / profile, JWT (access + refresh), bcrypt password
  hashing, strong password validation, RBAC (`admin`, `doctor`, `medical_student`).
  Public registration can never self-assign `admin` — see `create_admin.py`.
- **Uploads**: admin-only PDF upload with extension + magic-byte + size
  validation, files saved under generated (non-guessable) names.
- **RAG pipeline**: extract (PyMuPDF) → chunk (page-aware, overlapping) →
  embed (Sentence-Transformers) → store (FAISS, `IndexIDMap` keyed to
  Postgres `chunks.id`) → retrieve top 10 → cross-encoder rerank to top 5 →
  prompt the LLM → parse strict JSON → cite sources.
- **Chat**: ask questions, get an answer with confidence + citations +
  related documents; conversation history stored and retrievable.
- **Admin dashboard**: user/document/question counts, embedding status,
  top questions.
- **Security baked in throughout** (see below), not bolted on after.

## Security measures already in place

| Concern | Mitigation |
|---|---|
| Password storage | bcrypt via passlib, never plaintext |
| Auth | JWT with short-lived access tokens + refresh tokens |
| Privilege escalation | RBAC dependency (`require_roles`), admin can't be self-registered |
| User enumeration | Generic login error, dummy-hash check when user doesn't exist |
| File upload abuse | Extension + magic-byte + size checks, generated filenames, path traversal avoided |
| Prompt injection | Retrieved document text is wrapped in a delimited block with explicit "treat as data, not instructions" guidance |
| Hallucination | LLM instructed to only use provided context and say so if evidence is missing; app forces confidence to 0 if no chunks were retrieved |
| Malformed LLM output | Defensive JSON parsing with a fail-safe "insufficient evidence" fallback — never shows broken output to a clinician |
| Abuse / DoS | Rate limiting via slowapi on auth and chat endpoints |
| XSS/clickjacking | Security headers on every response (`X-Content-Type-Options`, `X-Frame-Options`, HSTS in prod) |
| CORS | Explicit origin allow-list, never `*` |
| IDOR | History delete/read scoped strictly to the authenticated user's own `user_id` |
| Secrets | `JWT_SECRET` has no default — app refuses to boot without it; `.env` gitignored |

## Running it locally

1. `cp .env.example .env` and fill in a real `JWT_SECRET`:
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(64))"
   ```
2. Start everything:
   ```bash
   docker compose up --build
   ```
3. Pull the LLM model into the Ollama container (matches `LLM_MODEL` in `.env`):
   ```bash
   docker compose exec ollama ollama pull mistral
   ```
4. Create the first admin account:
   ```bash
   docker compose exec backend python -m app.utils.create_admin
   ```
5. API is live at `http://localhost:8000` (docs at `/docs` in non-production `ENV`),
   or via Nginx at `http://localhost/api/`.

## Swapping the LLM

Nothing in the code references a specific model. Change `.env`:
```
LLM_MODEL=gemma2
```
then `docker compose exec ollama ollama pull gemma2` and restart the backend
container. Same goes for the embedding model (`EMBEDDING_MODEL`) — though if
you change its output dimension, update `_EMBEDDING_DIM` in
`app/rag/retriever.py` and rebuild the FAISS index from scratch.

## Remaining work before frontend development

- **Frontend** (React/Vite/Tailwind/Shadcn) — not built yet; the backend API
  (versioned at `/api/v1/...`) is ready to be consumed by it.
- **Prompt-injection classifier / stricter sanitization** — current
  mitigation is prompt-level instruction; a dedicated filter on extracted
  text (e.g. stripping suspicious imperative sentences before they ever
  reach the prompt) would harden this further.
- **Refresh-token revocation** — refresh tokens are issued but there's no
  server-side revocation store yet; add one (Redis is already in the stack)
  before treating this as fully production-hardened auth.
- **FAISS/BM25 horizontal scaling** — both are still local files behind a
  lock; this works for one backend replica, not multiple. Documented
  migration path is pgvector/Qdrant for FAISS and OpenSearch for BM25 if
  the corpus or traffic outgrows a single instance.
- **Prometheus/Grafana as running services** — the backend exposes
  `/metrics` correctly (request counts, per-stage RAG latency, cache hit
  ratio, active users), but no Prometheus/Grafana containers are wired into
  `docker-compose.yml` yet to actually scrape and visualize them.
- **CI/CD pipeline** — no GitHub Actions workflow exists yet for automated
  lint/test/build/security-scan on push.
- **Test coverage** — 76% and real (not estimated), but `app/rag/hybrid.py`,
  `app/evaluation/pipeline.py`, and `app/tasks/evaluation_tasks.py` remain
  under 35% — each needs FAISS/BM25/LLM mocking that's a real chunk of
  work, correctly scoped as follow-up rather than rushed.

Tell me which of these to build next and I'll pick up right where this leaves off.
