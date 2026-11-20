#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("get_ipython().run_line_magic('pip', 'install sickle')")
print("get_ipython().run_line_magic('pip', 'install sentence-transformers chromadb')")

# In[2]:


import json
from sickle import Sickle
import chromadb
from sentence_transformers import SentenceTransformer


# In[ ]:


sickle = Sickle('https://eprints2.undip.ac.id/cgi/oai2')

print("Memulai proses harvesting dan filtering data ber-abstrak...")

try:
    records = sickle.ListRecords(metadataPrefix='oai_dc')

    harvested_data = []
    target_count = 500  # Target data valid yang ingin dikumpulkan
    count = 0
    scanned = 0

    for record in records:
        scanned += 1

        # PENGAMAN UTAMA: Tangkap error jika record kosong atau tidak punya metadata (deleted record)
        try:
            metadata = record.metadata
        except AttributeError:
            continue  # Lewati record yang rusak/tidak punya metadata

        # Ekstraksi field dengan aman
        title = metadata.get('title', [''])[0]
        creators = metadata.get('creator', [])
        description = metadata.get('description', [''])[0] # Ini abstrak
        identifiers = metadata.get('identifier', [])
        dates = metadata.get('date', [])

        # Ambil tanggal pertama jika tersedia
        publication_date = dates[0] if dates else 'Tanggal tidak diketahui'

        # Cari URL dokumen asli
        url = next((i for i in identifiers if 'http' in i), '')

        # FILTER TAMBAHAN: Singkirkan dokumen administratif (lembar penilaian/peer review)
        is_administrative = "penilaian sejawat" in title.lower() or "peer_review" in url.lower()

        # Syarat valid: Deskripsi ada, panjang > 50 karakter, dan bukan dokumen administratif
        if description and len(description.strip()) > 50 and not is_administrative:
            item = {
                "title": title,
                "creators": creators,
                "description": description,
                "url": url,
                "date": publication_date
            }

            harvested_data.append(item)
            count += 1
            print(f"[Valid #{count}] (Scan ke-{scanned}) Dapatkan: {title[:50]}...")

            # Berhenti jika target sudah tercapai
            if count >= target_count:
                break

    # Simpan hasil panen yang sudah bersih ke file JSON
    with open('eprints_undip_clean.json', 'w', encoding='utf-8') as f:
        json.dump(harvested_data, f, ensure_ascii=False, indent=4)

    print(f"\nSelesai! Berhasil mengamankan {len(harvested_data)} data valid ke 'eprints_undip_clean.json'.")

except Exception as e:
    print(f"Terjadi kendala umum saat koneksi/harvesting: {e}")


# In[4]:


print("1. Memuat model AI Embedding (multilingual)...")
# Model ini sangat bagus untuk bahasa Indonesia & Inggris secara bersamaan
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

print("2. Menghubungkan ke ChromaDB (database vektor lokal)...")
client = chromadb.PersistentClient(path="./eprints_vector_db")

# Hapus koleksi lama jika ingin memperbarui dari awal
try:
    client.delete_collection(name="undip_research_search")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="undip_research_search",
    metadata={"hnsw:space": "cosine"} # Mengubah metrik jarak menjadi Cosine
)

print("3. Membaca data dari file JSON hasil panen...")
with open('eprints_undip_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

documents = []
metadatas = []
ids = []

for idx, item in enumerate(data):
    # Gabungkan Judul dan Deskripsi/Abstrak agar konteks pencariannya makin kaya
    text_to_embed = f"{item['title']}. {item['description']}"

    doc_id = f"doc_{idx}"
    documents.append(text_to_embed)

    # Ambil tanggal dengan aman (mengantisipasi list atau string kosong)
    raw_date = item.get("date", "")
    date_str = raw_date[0] if isinstance(raw_date, list) and raw_date else str(raw_date)

    metadatas.append({
        "title": item["title"],
        "creators": ", ".join(item["creators"]) if isinstance(item["creators"], list) else str(item["creators"]),
        "url": item["url"],
        "description": item["description"],
        "date": date_str
    })
    ids.append(doc_id)

print("4. Melakukan embedding massal secara kilat (Batch Encoding)...")
# Menggunakan batch_size=64 dengan progress bar agar proses 10.000 data cepat dan terpantau
embeddings = model.encode(documents, batch_size=64, show_progress_bar=True).tolist()

print("5. Memasukkan vektor dan metadata ke ChromaDB secara bertahap (Batch 1000)...")
batch_size = 1000  # Batas aman di bawah limit maksimal ChromaDB (5461)
total_data = len(ids)

for i in range(0, total_data, batch_size):
    end_idx = min(i + batch_size, total_data)

    # Masukkan per batch ke database
    collection.add(
        ids=ids[i:end_idx],
        embeddings=embeddings[i:end_idx],
        metadatas=metadatas[i:end_idx],
        documents=documents[i:end_idx]
    )
    print(f"Progress: Berhasil menyimpan data ke-{i} sampai {end_idx}...")

print(f"\nSukses! Seluruh {total_data} dokumen berhasil di-vektorisasi dan disimpan ke Vector Database.")


# In[5]:


# Load model & database yang sudah dibuat
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
client = chromadb.PersistentClient(path="./eprints_vector_db")
collection = client.get_collection(name="undip_research_search")

# Simulasi user mengetik query pencarian dengan bahasa bebas / konsep (bukan keyword pas)
query = "penelitian tentang kemiskinan"
MAX_DISTANCE_THRESHOLD = 0.5
print(f"Query User: '{query}'\n" + "="*50)

# Ubah query user jadi vektor
query_vector = model.encode(query).tolist()

# Cari 3 dokumen yang paling mirip maknanya di database
results = collection.query(
    query_embeddings=[query_vector],
    n_results=100,
    include=['metadatas', 'distances']
)

json_results = []

for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
    # FILTER THRESHOLD: Lewati dokumen jika jaraknya melebihi batas maksimal
    if distance > MAX_DISTANCE_THRESHOLD:
        continue # Skip dokumen yang tidak memenuhi syarat relevansi

    # Di ChromaDB (dengan Cosine distance), nilai distance yang SEMAKIN KECIL (mendekati 0.0) 
    # artinya SEMAKIN MIRIP/RELEVAN.
    print(f"Peringkat #{i+1} (Skor Jarak: {distance:.4f})")
    print(f"Judul   : {metadata['title']}")
    print(f"Penulis : {metadata['creators']}")
    print(f"url: {metadata['url']}")
    print(f"date: {metadata['date']}")
    print("-" * 60)

    json_results.append({
        "peringkat": i+1,
        "skor": distance,
        "judul": metadata['title'],
        "penulis": metadata['creators'],
        "deskripsi": metadata['description'],
        "url": metadata['url'],
        "date": metadata['date'] 
    })

print(json_results)

