# Data Pipeline

How the vector database gets built, and the contract the service depends on.
Everything here happens in `data_harvesting.ipynb` — there is no other ingestion
path.

> The app reads `./eprints_vector_db` and refuses to start without it. That
> directory and `eprints_undip_clean.json` are git-ignored, so a fresh clone has
> no data until you run the notebook.

## Pipeline overview

```
OAI-PMH (v.undip.ac.id/cgi/oai2)
        │  harvest + filter
        ▼
eprints_undip_clean.json          (~1,000 records)
        │  embed "{title}. {description}"
        ▼
eprints_vector_db/                 ChromaDB, collection "undip_research_search"
```

## Stage 1 — Harvest

| Setting | Value |
|---|---|
| Endpoint | `https://v.undip.ac.id/cgi/oai2` |
| Metadata format | `oai_dc` |
| Client | `sickle` |
| Target | 1,000 valid records |
| Observed | 1,000 kept out of 5,035 scanned |

A record is kept when **all** of these hold:

- it has metadata (deleted/broken records are skipped);
- it has a `description` whose trimmed length exceeds 50 characters;
- the title does not contain `penilaian sejawat` and the URL does not contain
  `peer_review` (those are administrative review sheets, not papers).

Extraction rules:

| Field | Rule |
|---|---|
| `title` | first `dc:title` |
| `creators` | all `dc:creator` values (a list) |
| `description` | first `dc:description` — treated as the abstract |
| `url` | first `dc:identifier` containing `http` |
| `date` | first `dc:date`, else `"Tanggal tidak diketahui"` |

Result is written to `eprints_undip_clean.json`.

Because harvesting stops at 1,000, the indexed corpus is an **early-page sample**
of the OAI feed, not a random or complete sample of Undip's repository.

## Stage 2 — Embed and index

| Setting | Value |
|---|---|
| Model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector size | 384 |
| Text embedded | `"{title}. {description}"` |
| Encoding | `batch_size=64` |
| Collection | `undip_research_search` |
| Distance | `{"hnsw:space": "cosine"}` |
| Insert batch | 1,000 (Chroma rejects batches above ~5,461) |
| Document IDs | `doc_0`, `doc_1`, … |

**The notebook deletes the collection first**, then recreates it. Re-running it
wipes every vector; there is no incremental update path. Stop the API server
before re-ingesting so it is not querying a collection that is being replaced.

The model truncates at 128 tokens, so a stored vector effectively represents the
title plus the opening of the abstract — not the whole document.

## Metadata contract

The notebook owns the metadata shape; `backend.py` reads it. Change both together.

| Chroma metadata key | Source | Returned as | Notes |
|---|---|---|---|
| `title` | `dc:title` | `judul` | Missing → `""` |
| `creators` | `dc:creator`, joined with `", "` | `penulis` | Always a string, never a list |
| `url` | first `http` identifier | `url` | May be `""` |
| `description` | `dc:description` | `deskripsi` | Also part of the embedded text |
| `date` | `dc:date` | `date` | Free text, not parsed as a date |

The service tolerates missing keys (they become `""`) but not a changed distance
metric: if the collection was not created with `hnsw:space=cosine`, `skor` is not
a similarity. `/health` reports the metric so this is checkable.

## Rebuilding from scratch

1. Stop the API server.
2. Run `data_harvesting.ipynb` top to bottom:
   - cell 0 installs `sickle`, `sentence-transformers`, `chromadb`;
   - cell 2 harvests to `eprints_undip_clean.json`;
   - cell 3 deletes/recreates the collection and loads it;
   - cell 4 is an interactive sanity check only. It filters by **distance**
     (`distance <= 0.5`), the opposite of the API's minimum-similarity threshold,
     so do not copy its logic into a client.
3. Start the server, then confirm:
   ```bash
   curl http://localhost:8000/health
   # documents == number of harvested records, metric == "cosine"
   ```
4. Calibrate the threshold against the new data:
   ```bash
   python calibrate_threshold.py
   ```

`requirements.txt` does **not** include `sickle` — it is installed by the
notebook's first cell and is only needed for harvesting.

## Reproducibility notes

- The harvested JSON and the vector DB are both git-ignored. Rebuilding requires
  network access to the Undip OAI endpoint, which is outside this repository's
  control.
- The notebook was executed on Python 3.14 while the service runs on Python 3.11
  (Docker). This does not change the vectors, but it means the notebook was not
  validated against the runtime interpreter.
- The embedding model is referenced by name, not by revision. An upstream model
  update would silently produce vectors that no longer match queries.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: sickle` | Notebook cell 0 not run | Run it, or `pip install sickle` |
| Harvest prints an exception and stops | OAI endpoint unreachable or rate-limited | Retry later; check the endpoint in a browser |
| Fewer than 1,000 records saved | The OAI stream ended early, or the loop hit an exception | Read the printed error and re-run; the count is not a filter problem |
| `eprints_undip_clean.json` missing after an error | The whole harvest is wrapped in a broad `try/except` | Re-run the cell; the file is only written on success |
| Chroma rejects an insert batch | Batch over the ~5,461 cap | Keep the 1,000-record batching |
| `documents: 0` in `/health` after ingest | Wrong `DB_PATH` (relative to CWD) | Run the notebook and the server from the repo root |
| Search results look random | Collection not built with `hnsw:space=cosine` | Recreate the collection (cell 3) |
| Docker sees no data | `./eprints_vector_db` missing on the host | Build it on the host; compose bind-mounts it |
