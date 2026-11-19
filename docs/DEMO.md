# Demo Runbook

Everything you need to walk into the meeting with a working demo and a plan for
when something does not cooperate. Written for the prototype as it exists today.

## The day before

- [ ] `cp .env.example .env`, then fill `GEMINI_API_KEY` and `GEMINI_MODEL`.
  Confirm the model name is currently available — a retired name does not break
  startup, it silently downgrades every answer to `list_only`.
- [ ] Confirm `./eprints_vector_db` exists on **the laptop you will present
  from**. If you built it on another machine, copy the whole directory.
- [ ] Start the app once so the embedding model downloads:
  ```bash
  uvicorn backend:app --reload --port 8000 --env-file .env
  ```
  `--env-file` is what actually feeds `GEMINI_API_KEY` / `GEMINI_MODEL` to a local
  run — without it the API quietly degrades to `list_only`.
  The first start takes minutes; later starts take seconds. Do not clear the
  HuggingFace cache. If you demo via Docker, use the same machine and keep the
  `model_cache` volume.
- [ ] Check all three health facts in one call:
  ```bash
  curl http://localhost:8000/health
  ```
  Expect `"documents": 1000` (or whatever your harvest produced),
  `"llm_configured": true`, `"metric": "cosine"`.
- [ ] Calibrate the threshold:
  ```bash
  python calibrate_threshold.py
  ```
  Paste the suggested `THRESHOLD_DEFAULT` into `backend.py`, restart, and check
  that a normal query returns `"status": "full"`.
- [ ] Run one full query end to end **after** that change. A response with
  `"status": "full"` and a non-empty `ringkasan` is the only proof that the LLM
  path works.
- [ ] Take a screenshot of a successful `full` response.
- [ ] Decide what the audience sees: Swagger UI at `/docs` (zero setup, shows the
  whole contract, good for a technical audience) or the frontend.

## On the day

- [ ] Start the server:
  ```bash
  uvicorn backend:app --port 8000 --env-file .env
  ```
  Prefer this over `docker compose` — fewer moving parts and the model cache is
  already warm.
- [ ] Warm up with one request before anyone is watching (first request loads
  nothing new, but it shakes out any environment surprise):
  ```bash
  curl -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{"query":"kemiskinan","threshold":0.5}'
  ```
- [ ] Keep a terminal with the expected curls ready; do not type JSON live.

## Suggested queries

| Purpose | Query | Expected |
|---|---|---|
| Main demo (Indonesian) | `faktor risiko stunting pada balita` | `full`, several papers, summary with `[1]` citations |
| Cross-language (English) | `machine learning for sentiment analysis` | `full` — shows the model is multilingual |
| Deliberate `no_results` | `faktor risiko stunting` with `"threshold": 0.85` | `no_results`, `skor_tertinggi` around 0.4–0.6, `arahan` telling you to lower the threshold |

Show the third one on purpose: it demonstrates that the API explains itself
instead of looking broken, and that the threshold is a real, tunable knob.

## If something goes wrong

| Situation | What to say / do | Fallback |
|---|---|---|
| Model starts downloading | "It caches the model on first run" | Wait; it is a one-time cost |
| `status=list_only` with a quota message | "The search still works; the AI layer is rate-limited" | Show the ranked list — that is the fallback design, not a failure |
| `status=list_only` with "belum dikonfigurasi" | `.env` not loaded — usually a missing `--env-file .env` | Restart with the flag |
| Empty results on everything | Threshold is too strict, or DB is missing | Lower the threshold live; check `/health` |
| Server will not start (`collection ... was not found`) | Wrong directory or missing DB | Start uvicorn from the repo root; if the DB is missing, switch to screenshots |
| Network is down | Gemini is unreachable | Screenshots of a `full` response; the retrieval path works offline |

## Q&A crib sheet

Answers that are honest and short. Do not overclaim.

- **How many papers?** 1,000 indexed, selected from about 5,035 records scanned in
  the OAI feed — an early-page sample, not a random one.
- **How does search work?** The query and each paper (title + abstract) are
  embedded with the same multilingual model; we rank by cosine similarity and
  apply a minimum-similarity threshold.
- **Why do you truncate abstracts?** The model handles 128 tokens, so vectors
  represent the title plus the start of the abstract. A production version would
  chunk or summarize first.
- **Why is the threshold 0.5?** It is a placeholder; `calibrate_threshold.py`
  measures real scores and suggests a value. If the demo threshold looks
  arbitrary, that is exactly what the script fixes.
- **What if Gemini fails?** The response degrades to `list_only` with a reason in
  `ringkasan_error`; searching never depends on the LLM. There is no automatic
  retry — the user retries.
- **Can the summary invent sources?** Every claim is supposed to carry `[n]`
  matching the retrieved paper's rank, and citations pointing outside the
  retrieved set are stripped before the response is returned.
- **Why no login / rate limit?** Prototype scope; it is a local demo, and the
  service has no user accounts or stored data.
- **What would you do next?** Calibrate the threshold, index the full corpus,
  add pagination, pin the model revision, and add auth before exposing it.
