"""Setel THRESHOLD_DEFAULT di backend.py memakai data asli.

Threshold adalah kemiripan minimum (1 - jarak cosine). Nilai awalnya 0.5
masih tebakan: kalau ternyata dokumen yang relevan hanya berskor ~0.35,
semua query demo akan berakhir `no_results`.

Jalankan dari root repo (butuh ./eprints_vector_db):

    python calibrate_threshold.py
    python calibrate_threshold.py "query satu" "query dua"

Catatan: model embedding dimuat di sini juga, jadi jalankan sekali sebelum
demo supaya cache-nya hangat.
"""

import sys

import chromadb
from sentence_transformers import SentenceTransformer

from backend import (
    COLLECTION_NAME,
    DB_PATH,
    MODEL_NAME,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
)

SAMPLE_QUERIES = [
    "penelitian tentang kemiskinan",
    "pengaruh media sosial terhadap remaja",
    "analisis kualitas air sungai",
    "sistem informasi berbasis web",
    "faktor risiko stunting pada balita",
    "machine learning for sentiment analysis",
]

TOP_K = 5


def main(queries: list[str]) -> None:
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        sys.exit(
            f"Koleksi '{COLLECTION_NAME}' tidak ditemukan di '{DB_PATH}'.\n"
            "Jalankan notebook data_harvesting.ipynb lebih dulu."
        )

    total = collection.count()
    if total == 0:
        sys.exit(f"Koleksi '{COLLECTION_NAME}' kosong. Jalankan notebook dulu.")
    print(f"{total} dokumen terindeks di '{DB_PATH}'.\n")

    model = SentenceTransformer(MODEL_NAME)
    skor_top1: list[float] = []

    for query in queries:
        raw = collection.query(
            query_embeddings=[model.encode(query).tolist()],
            n_results=TOP_K,
            include=["metadatas", "distances"],
        )
        skor = [round(1.0 - d, 4) for d in raw["distances"][0]]
        skor_top1.append(skor[0])
        teratas = (raw["metadatas"][0][0] or {}).get("title", "(tanpa judul)")
        print(f"Query : {query}")
        print(f"  skor top-{TOP_K}: {skor}")
        print(f"  teratas    : {str(teratas)[:80]}")

    # Default yang aman untuk demo: di bawah skor terbaik query terlemah,
    # supaya tidak ada query yang mengembalikan hasil kosong.
    disarankan = round(min(skor_top1) - 0.05, 2)
    disarankan = max(THRESHOLD_MIN, min(THRESHOLD_MAX, disarankan))

    print("\n" + "=" * 60)
    print(f"Skor terbaik per query : {[round(s, 2) for s in skor_top1]}")
    print(f"Rentang skor terendah  : {min(skor_top1):.2f} .. {max(skor_top1):.2f}")
    print('\nSetel di backend.py:')
    print(f"    THRESHOLD_DEFAULT = {disarankan}")
    print(
        "\nPastikan hasil teratas memang relevan; threshold hanya mengatur "
        "seberapa ketat, bukan seberapa relevan."
    )


if __name__ == "__main__":
    main(sys.argv[1:] or SAMPLE_QUERIES)
