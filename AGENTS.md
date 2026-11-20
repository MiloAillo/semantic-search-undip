# AGENTS.md

## What this repo is

Monorepo root. The only service today is `backend/` — a FastAPI semantic search
over harvested Universitas Diponegoro (Undip) eprints. Future services (e.g. a
frontend) live as siblings of `backend/`.

```
backend/          the service: code, tests, docs, Docker, harvested data
AGENTS.md         this file (monorepo-level rules)
```

Service-specific instructions (architecture, ingestion, API contract, data
prerequisites) are in [`backend/AGENTS.md`](backend/AGENTS.md). Read that before
changing anything under `backend/`.

## Where to run commands

All service paths are relative to `backend/`: `DB_PATH = "./eprints_vector_db"`,
`pytest.ini` (`pythonpath = .`), and `--env-file .env` all assume that directory
as the working directory.

```bash
cd backend
pip install -r requirements.txt
uvicorn backend:app --reload --port 8000 --env-file .env
pytest
APP_UID=$(id -u) docker compose up -d --build --wait
```

Running these from the monorepo root fails: `backend.py` will not be importable
and the vector DB will not be found.

## Conventions

- Do not commit secrets or generated data. `backend/.env`,
  `backend/eprints_vector_db/` and `backend/eprints_undip_clean.json` are
  git-ignored; `.env.example` documents the required variables.
- Keep the service self-contained under `backend/`; root-level files are for
  cross-service concerns only.
