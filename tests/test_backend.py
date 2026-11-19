import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

import backend


class FakeModel:
    def encode(self, text):
        return np.array([1.0, 0.0])


class FakeCollection:
    """`distances` = jarak cosine yang dikembalikan ChromaDB (urut naik)."""

    def __init__(self, distances):
        self.distances = distances

    def query(self, query_embeddings, n_results, include):
        metas = [
            {
                "title": f"Judul {i}",
                "creators": "A, B",
                "description": f"Abstrak {i}",
                "url": f"http://x/{i}.pdf",
                "date": "2021",
            }
            for i in range(len(self.distances))
        ]
        return {"metadatas": [metas], "distances": [self.distances]}

    def count(self):
        return len(self.distances)


def make_client(distances, summarize=None):
    # TestClient dipakai TANPA `with`, jadi lifespan (model + ChromaDB asli)
    # tidak dijalankan; komponen diganti dengan yang palsu.
    backend.app.state.model = FakeModel()
    backend.app.state.collection = FakeCollection(distances)
    backend.app.state.summarize = summarize or (lambda q, r: "Ringkasan [1]")
    return TestClient(backend.app)


# ---------- validasi input ----------
@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "kemiskinan", "threshold": backend.THRESHOLD_MAX + 0.01},
        {"query": "kemiskinan", "threshold": backend.THRESHOLD_MIN - 0.01},
        {},
    ],
)
def test_input_tidak_valid_ditolak(body):
    assert make_client([0.1]).post("/search", json=body).status_code == 422


def test_threshold_pada_batas_diterima():
    c = make_client([0.1])
    for t in (backend.THRESHOLD_MIN, backend.THRESHOLD_MAX):
        assert c.post("/search", json={"query": "x", "threshold": t}).status_code == 200


# ---------- threshold: makin tinggi makin ketat ----------
def test_threshold_lebih_tinggi_lebih_sedikit_hasil():
    c = make_client([0.1, 0.3, 0.5, 0.7])  # kemiripan 0.9, 0.7, 0.5, 0.3
    longgar = c.post("/search", json={"query": "x", "threshold": 0.3}).json()
    ketat = c.post("/search", json={"query": "x", "threshold": 0.7}).json()
    assert longgar["count"] == 4
    assert ketat["count"] == 2
    assert [r["skor"] for r in ketat["results"]] == [0.9, 0.7]


def test_kemiripan_sama_dengan_threshold_masih_lolos():
    # jarak 0.3 -> kemiripan 0.7; float 1-0.3 harus tetap lolos di threshold 0.7
    r = make_client([0.3]).post("/search", json={"query": "x", "threshold": 0.7})
    assert r.json()["count"] == 1


# ---------- tiga jalur respons ----------
def test_tidak_ada_jurnal_lolos_memberi_arahan():
    r = make_client([0.9, 0.8]).post("/search", json={"query": "x", "threshold": 0.5})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "no_results"
    assert body["count"] == 0 and body["results"] == []
    assert body["arahan"] and body["ringkasan"] is None


def test_hasil_lengkap_dengan_ringkasan():
    r = make_client([0.1, 0.2, 0.9]).post("/search", json={"query": "x", "threshold": 0.5})
    body = r.json()
    assert body["status"] == "full"
    assert body["count"] == 2
    assert body["ringkasan"] == "Ringkasan [1]"
    assert body["ringkasan_error"] is None
    assert [x["peringkat"] for x in body["results"]] == [1, 2]


def test_llm_gagal_fallback_dan_pesan_error():
    def broken(q, r):
        raise backend.LLMError("Kuota habis.")

    body = make_client([0.1, 0.2], summarize=broken).post(
        "/search", json={"query": "x"}
    ).json()
    assert body["status"] == "list_only"
    assert body["ringkasan"] is None
    assert body["ringkasan_error"] == "Kuota habis."
    assert body["count"] == 2


def test_error_tak_terduga_tidak_membocorkan_detail():
    def boom(q, r):
        raise ValueError("rahasia internal")

    body = make_client([0.1], summarize=boom).post("/search", json={"query": "x"}).json()
    assert body["status"] == "list_only"
    assert "rahasia" not in body["ringkasan_error"]


def test_llm_timeout(monkeypatch):
    monkeypatch.setattr(backend, "LLM_TIMEOUT_SECONDS", 0.05)

    def lambat(q, r):
        time.sleep(0.3)
        return "terlambat"

    body = make_client([0.1], summarize=lambat).post("/search", json={"query": "x"}).json()
    assert body["status"] == "list_only"
    assert "tepat waktu" in body["ringkasan_error"]


def test_llm_hanya_menerima_top_k():
    received = {}

    def spy(query, results):
        received["n"] = len(results)
        return "ok"

    make_client([0.01 * i for i in range(20)], summarize=spy).post(
        "/search", json={"query": "x", "threshold": 0.5}
    )
    assert received["n"] == backend.SUMMARY_TOP_K


# ---------- LLM: konfigurasi, error, sitasi ----------
def test_llm_belum_dikonfigurasi(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(backend.LLMError, match="belum dikonfigurasi"):
        backend.summarize_with_llm("x", [])


class _FakeModels:
    def __init__(self, behavior):
        self.behavior = behavior

    def generate_content(self, model, contents):
        return self.behavior()


class _FakeClient:
    def __init__(self, behavior):
        self.models = _FakeModels(behavior)


def _patch_client(monkeypatch, behavior):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    from google import genai

    monkeypatch.setattr(genai, "Client", lambda api_key: _FakeClient(behavior))


def _hasil(n):
    return [
        backend.SearchResult(
            peringkat=i, skor=0.9, judul=f"J{i}", penulis="A",
            deskripsi="d", url="u", date="2021",
        )
        for i in range(1, n + 1)
    ]


def test_llm_429_jadi_pesan_kuota(monkeypatch):
    def raise_429():
        raise genai_errors.APIError(429, {"error": {"message": "quota"}})

    _patch_client(monkeypatch, raise_429)
    with pytest.raises(backend.LLMError, match="Kuota"):
        backend.summarize_with_llm("x", _hasil(2))


def test_llm_teks_kosong_dianggap_gagal(monkeypatch):
    class Resp:
        text = "   "

    _patch_client(monkeypatch, lambda: Resp())
    with pytest.raises(backend.LLMError, match="tidak menghasilkan"):
        backend.summarize_with_llm("x", _hasil(2))


def test_llm_sukses_dan_sitasi_tidak_valid_dibuang(monkeypatch):
    class Resp:
        text = "Kemiskinan menurun [1][2]. Klaim lain [7]."

    _patch_client(monkeypatch, lambda: Resp())
    out = backend.summarize_with_llm("x", _hasil(2))
    assert "[1]" in out and "[2]" in out
    assert "[7]" not in out


# ---------- lain-lain ----------
def test_health():
    assert make_client([0.1, 0.2, 0.3]).get("/health").json() == {
        "status": "ok",
        "documents": 3,
    }


def test_startup_gagal_jelas_bila_koleksi_tidak_ada(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "DB_PATH", str(tmp_path / "kosong"))
    with pytest.raises(RuntimeError, match="was not found"):
        with TestClient(backend.app):
            pass
