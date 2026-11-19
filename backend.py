import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Literal

import chromadb
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

load_dotenv()

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "./eprints_vector_db"
COLLECTION_NAME = "undip_research_search"
N_RESULTS = 100

# Threshold = kemiripan minimum (1 - jarak cosine). Makin tinggi = makin ketat.
# Batas bawah/atas slider UI. NILAI AWAL, setel setelah mencoba data asli.
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_DEFAULT = 0.5

# Jumlah jurnal teratas yang dikirim ke LLM (angka awal, bisa disetel tim).
SUMMARY_TOP_K = 5
# Batas waktu menunggu LLM sebelum dianggap gagal (detik).
LLM_TIMEOUT_SECONDS = 20

# Isi "arahan praktis" saat tidak ada jurnal yang lolos threshold.
GUIDANCE = [
    "Turunkan threshold agar hasil lebih longgar.",
    "Coba kata kunci lain.",
    "Perluas topik pencarian.",
]

# Logging dasar untuk observability. Di bawah uvicorn, logger aplikasi
# tidak otomatis aktif, jadi dikonfigurasi di sini.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("aisen.search")


# Validasi startup (dari backend.py versi tim, dipertahankan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        app.state.collection = client.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError as exc:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' was not found "
            f"in '{DB_PATH}'. Check the database path and make sure "
            "you've run the data-ingestion step."
        ) from exc

    # Import dipindah ke sini agar test yang tidak butuh model
    # tidak perlu memuat torch.
    from sentence_transformers import SentenceTransformer

    app.state.model = SentenceTransformer(MODEL_NAME)
    app.state.summarize = summarize_with_llm  # mudah diganti saat test
    if not (os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_MODEL")):
        log.warning(
            "GEMINI_API_KEY / GEMINI_MODEL belum diatur: ringkasan AI nonaktif, "
            "API hanya mengembalikan daftar jurnal."
        )
    yield


app = FastAPI(
    title="Undip Research Semantic Search API",
    description="Semantic search over the harvested Undip eprints repository.",
    version="0.2.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class SearchRequest(BaseModel):
    # Spasi di awal/akhir dibuang, jadi query "   " ditolak (min_length=1).
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, description="Free-text search query (Indonesian or English).")
    max_threshold: float = Field(
        default=THRESHOLD_DEFAULT,
        validation_alias=AliasChoices("max_threshold", "threshold"),
        serialization_alias="max_threshold",
        ge=THRESHOLD_MIN,
        le=THRESHOLD_MAX,
        description="Minimum similarity (1 - cosine distance). Higher = stricter.",
    )


class SearchResult(BaseModel):
    peringkat: int
    skor: float  # BERUBAH: kemiripan (makin tinggi makin mirip), bukan jarak
    judul: str
    penulis: str
    deskripsi: str
    url: str
    date: str


class SearchResponse(BaseModel):
    query: str
    max_threshold: float
    count: int
    results: list[SearchResult]
    # Field tambahan:
    #   full       -> ringkasan AI + daftar jurnal
    #   list_only  -> LLM gagal, daftar jurnal saja (alasannya di `ringkasan_error`)
    #   no_results -> tidak ada jurnal lolos threshold, lihat `arahan`
    status: Literal["full", "list_only", "no_results"]
    ringkasan: str | None = None  # sitasi berformat [n], n = `peringkat`
    ringkasan_error: str | None = None  # pesan untuk pengguna; coba lagi manual
    arahan: list[str] = []


# ---- LLM (Google Gemini API) -----------------------------------------------
class LLMError(Exception):
    """Pesan exception ini ditampilkan ke pengguna, jadi tulis dalam Indonesia."""


def build_prompt(query: str, results: list[SearchResult]) -> str:
    sources = "\n\n".join(f"[{r.peringkat}] {r.judul}\n{r.deskripsi}" for r in results)
    return (
        "Anda adalah asisten riset. Tulis ringkasan singkat dalam bahasa Indonesia "
        "tentang isi, temuan utama, atau pola yang muncul dari jurnal yang disajikan. "
        "Gunakan HANYA sumber di bawah ini; jangan menambah fakta atau angka. "
        "Panjang maksimal 3 paragraf dan 180 kata. Utamakan poin paling relevan, "
        "hindari pengantar panjang dan jangan mengulang judul jurnal. "
        "Setiap klaim faktual wajib memiliki sitasi [n] sesuai nomor sumber. "
        "Gunakan Markdown sederhana hanya untuk penekanan penting (**teks**) bila perlu. "
        "Jika sumber tidak cukup, katakan secara singkat.\n\n"
        f"Pertanyaan pengguna: {query}\n\nSumber jurnal:\n{sources}"
    )


_CITATION = re.compile(r"\[(\d+)\]")


def sanitize_citations(text: str, valid: set[int]) -> str:
    """Buang sitasi [n] yang nomornya tidak ada di sumber yang dikirim ke LLM."""

    def keep(m: re.Match) -> str:
        if int(m.group(1)) in valid:
            return m.group(0)
        log.warning("Sitasi tidak valid dibuang: %s", m.group(0))
        return ""

    return _CITATION.sub(keep, text)


def summarize_with_llm(query: str, results: list[SearchResult]) -> str:
    """Panggil Gemini. Tanpa retry otomatis: gagal -> LLMError, pengguna
    mencoba lagi secara manual."""
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not (api_key and model):
        raise LLMError("Ringkasan AI belum dikonfigurasi.")

    from google import genai
    from google.genai import errors as genai_errors

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model, contents=build_prompt(query, results)
        )
    except genai_errors.APIError as exc:
        if exc.code == 429:
            raise LLMError(
                "Kuota atau batas permintaan LLM tercapai. Coba lagi sebentar lagi."
            ) from exc
        raise LLMError(
            f"Layanan LLM mengembalikan error ({exc.code}). Coba lagi."
        ) from exc

    text = (resp.text or "").strip()
    if not text:
        raise LLMError("LLM tidak menghasilkan ringkasan. Coba lagi.")
    return sanitize_citations(text, {r.peringkat for r in results})


@app.get("/health")
def health() -> dict:
    # Untuk pengecekan hidup/mati dan jumlah dokumen ter-index.
    return {"status": "ok", "documents": app.state.collection.count()}


@app.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    model = app.state.model
    collection = app.state.collection

    query_vector = await run_in_threadpool(model.encode, payload.query)
    query_vector = query_vector.tolist()

    raw = await run_in_threadpool(
        collection.query,
        query_embeddings=[query_vector],
        n_results=N_RESULTS,
        include=["metadatas", "distances"],
    )

    matched: list[tuple[float, dict]] = []
    for metadata, distance in zip(raw["metadatas"][0], raw["distances"][0]):
        similarity = round(1.0 - distance, 6)
        if similarity >= payload.max_threshold:
            matched.append((distance, metadata))

    matched.sort(key=lambda item: item[0])
    results = [
        SearchResult(
            peringkat=rank,
            skor=round(1.0 - distance, 6),
            judul=metadata["title"],
            penulis=metadata["creators"],
            deskripsi=metadata["description"],
            url=metadata["url"],
            date=metadata["date"],
        )
        for rank, (distance, metadata) in enumerate(matched, start=1)
    ]

    # Cabang flowchart setelah threshold
    ringkasan = None
    ringkasan_error = None
    arahan: list[str] = []
    if not results:
        status = "no_results"
        arahan = GUIDANCE
    else:
        try:
            ringkasan = await asyncio.wait_for(
                run_in_threadpool(
                    app.state.summarize, payload.query, results[:SUMMARY_TOP_K]
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            status = "full"
        except asyncio.TimeoutError:
            log.warning("LLM timeout setelah %ss", LLM_TIMEOUT_SECONDS)
            status = "list_only"
            ringkasan_error = "LLM tidak merespons tepat waktu. Coba lagi."
        except LLMError as exc:
            log.warning("LLM gagal: %s", exc)
            status = "list_only"
            ringkasan_error = str(exc)
        except Exception:
            log.exception("LLM gagal (error tak terduga)")
            status = "list_only"
            ringkasan_error = "Ringkasan gagal dibuat. Coba lagi."

    log.info(
        "search status=%s count=%d threshold=%.2f duration_ms=%.0f",
        status,
        len(results),
        payload.max_threshold,
        (time.perf_counter() - started) * 1000,
    )

    return SearchResponse(
        query=payload.query,
        max_threshold=payload.max_threshold,
        count=len(results),
        results=results,
        status=status,
        ringkasan=ringkasan,
        ringkasan_error=ringkasan_error,
        arahan=arahan,
    )


if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000)
