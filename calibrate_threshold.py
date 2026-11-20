"""Setel THRESHOLD_DEFAULT di backend.py memakai data asli.

Threshold adalah kemiripan minimum (1 - jarak cosine). Threshold bukan alat
untuk "memastikan selalu ada hasil": ia penjaga relevansi, dan hasil kosong
adalah jawaban yang sah.

Karena itu skrip ini tidak menyarankan angka yang menjamin setiap query
mengembalikan sesuatu. Ia memisahkan dua kelompok query:

  * RELEVAN - seharusnya mendapat hasil di korpus ini.
  * NOISE   - di luar topik, seharusnya tidak mendapat hasil.

Ambang hanya disarankan bila kedua kelompok terpisah, yaitu skor tertinggi
kelompok noise masih di bawah skor terendah kelompok relevan. Titik tengah
celah itulah yang dicetak. Bila kelompoknya tumpang tindih, tidak ada angka
yang bisa memisahkan relevan dari noise, dan skrip menolak menyarankan angka.

Jalankan dari root repo (butuh ./eprints_vector_db):

    python calibrate_threshold.py
    python calibrate_threshold.py "query satu" "query dua"

Argumen yang diberikan menggantikan daftar RELEVAN; daftar NOISE tetap dipakai.

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
    N_RESULTS,
    THRESHOLD_DEFAULT,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
)

# Query yang seharusnya mendapat hasil di korpus eprints2 Undip.
RELEVANT_QUERIES = [
    "analisis kualitas air sungai",
    "faktor risiko stunting pada balita",
    "pengaruh media sosial terhadap perilaku remaja",
    "sistem informasi berbasis web",
    "pemberdayaan masyarakat pesisir",
]

# Query di luar topik. Harus tidak mendapat hasil pada ambang yang dipilih.
NOISE_QUERIES = [
    "resep rendang padang",
    "jadwal pertandingan sepak bola liga inggris",
    "harga tiket konser musik jakarta",
]

TOP_K = 5


def skor_query(collection, model, query: str) -> tuple[list[float], str, int]:
    """Kembalikan (skor top-K, judul teratas, jumlah lolos threshold default)."""
    raw = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=N_RESULTS,
        include=["metadatas", "distances"],
    )
    distances = raw["distances"][0]
    skor = [round(1.0 - d, 4) for d in distances]
    teratas = (raw["metadatas"][0][0] or {}).get("title", "(tanpa judul)")
    lolos = sum(1 for s in skor if s >= THRESHOLD_DEFAULT)
    return skor[:TOP_K], str(teratas), lolos


def cetak_kelompok(collection, model, label: str, queries: list[str]) -> list[float]:
    print(f"\n{label}")
    print("-" * 64)
    skor_top1: list[float] = []
    for query in queries:
        skor, teratas, lolos = skor_query(collection, model, query)
        skor_top1.append(skor[0])
        print(f"  {query}")
        print(f"    top-{TOP_K}: {skor}")
        print(f"    teratas : {teratas[:70]}")
        print(f"    lolos @{THRESHOLD_DEFAULT:.2f}: {lolos} dokumen")
    return skor_top1


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
    print(f"{total} dokumen terindeks di '{DB_PATH}'.")

    model = SentenceTransformer(MODEL_NAME)

    relevan = queries or RELEVANT_QUERIES
    skor_relevan = cetak_kelompok(collection, model, "KELOMPOK RELEVAN", relevan)
    skor_noise = cetak_kelompok(collection, model, "KELOMPOK NOISE", NOISE_QUERIES)

    terlemah_relevan = min(skor_relevan)
    terkuat_noise = max(skor_noise)

    print("\n" + "=" * 64)
    print(f"Skor terendah query relevan : {terlemah_relevan:.2f}")
    print(f"Skor tertinggi query noise  : {terkuat_noise:.2f}")

    if terkuat_noise >= terlemah_relevan:
        print(
            "\nTIDAK ADA AMBANG YANG MEMISAHKAN.\n"
            f"Query noise mencapai {terkuat_noise:.2f}, sedangkan query relevan "
            f"terendah hanya {terlemah_relevan:.2f}.\n"
            "Angka apa pun di rentang ini akan meloloskan noise, atau membuang "
            "query relevan. Perbaiki dulu retrieval-nya (kualitas abstrak, "
            "model embedding, atau rumusan query), bukan threshold-nya."
        )
        print(f"\nTHRESHOLD_DEFAULT sekarang: {THRESHOLD_DEFAULT} (tidak diubah)")
        return

    disarankan = (terlemah_relevan + terkuat_noise) / 2
    disarankan = round(max(THRESHOLD_MIN, min(THRESHOLD_MAX, disarankan)), 2)
    print(f"\nCelah tersedia: {terkuat_noise:.2f} < threshold < {terlemah_relevan:.2f}")
    print("Setel di backend.py:")
    print(f"    THRESHOLD_DEFAULT = {disarankan}")
    print(
        "\nHasil kosong tetap sah: threshold menjaga relevansi, bukan memastikan "
        "setiap query terisi."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
