"""
EKSPLORASI FITUR GEMINI -- File Tunggal
==========================================
File ini menyatukan eksplorasi yang sebelumnya terpisah/terduplikasi di
app.py (fungsi eksplorasi_gemini) dan Exploring_gemini.py. Jalankan file
ini SENDIRI untuk eksperimen -- TIDAK perlu lagi dipanggil dari app.py
saat server start (sebelumnya app.py menjalankan eksplorasi_gemini() di
__main__, ini sebaiknya dipisah dari startup server produksi).

Untuk tiap eksperimen, INPUT dan OUTPUT dicetak eksplisit (revisi dosen:
"diperjelas input output setiap fiturnya").
"""

import os
import json
import sqlite3
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
DB_NAME = "hukum_rag.db"
MAX_RETRY = 5
JEDA_RETRY = 5  # detik


def panggil_dengan_retry(fungsi_api, **kwargs):
    """Retry untuk menangani error 503 (model Gemini sedang high demand)."""
    for percobaan in range(1, MAX_RETRY + 1):
        try:
            return fungsi_api(**kwargs)
        except Exception as e:
            print(f"    [RETRY {percobaan}/{MAX_RETRY}] Gagal: {e}")
            if percobaan < MAX_RETRY:
                time.sleep(JEDA_RETRY)
    raise RuntimeError(f"Gagal memanggil API Gemini setelah {MAX_RETRY} percobaan.")

TEKS_PASAL_UJI = (
    "Pasal 27 ayat (1): Setiap Orang dengan sengaja dan tanpa hak "
    "menyiarkan, mempertunjukkan, mendistribusikan, mentransmisikan, "
    "dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau "
    "Dokumen Elektronik yang memiliki muatan yang melanggar kesusilaan."
)
KUERI_UJI = "hukuman sebar konten asusila"


def cetak_header(judul):
    print("\n" + "=" * 70)
    print(judul)
    print("=" * 70)


def eksperimen_1_task_type():
    """
    Menguji apakah task_type mengubah representasi vektor untuk TEKS YANG SAMA.
    INPUT  : satu teks pasal yang sama, dikirim dengan 2 task_type berbeda
    OUTPUT : dua vektor berbeda meski input identik -> membuktikan model
             "mengarahkan" makna vektor sesuai tujuan penggunaannya
    """
    cetak_header("EKSPERIMEN 1: Variasi task_type pada teks yang SAMA")
    print(f"INPUT (sama untuk keduanya): \"{TEKS_PASAL_UJI[:80]}...\"\n")

    embed_doc = panggil_dengan_retry(
        client.models.embed_content,
        model="gemini-embedding-001",
        contents=TEKS_PASAL_UJI,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    embed_query = panggil_dengan_retry(
        client.models.embed_content,
        model="gemini-embedding-001",
        contents=TEKS_PASAL_UJI,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    embed_sim = panggil_dengan_retry(
        client.models.embed_content,
        model="gemini-embedding-001",
        contents=TEKS_PASAL_UJI,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )

    vd, vq, vs = embed_doc.embeddings[0].values, embed_query.embeddings[0].values, embed_sim.embeddings[0].values

    print(f"OUTPUT task_type=RETRIEVAL_DOCUMENT  : nilai[0]={vd[0]:.6f}, dimensi={len(vd)}")
    print(f"OUTPUT task_type=RETRIEVAL_QUERY     : nilai[0]={vq[0]:.6f}, dimensi={len(vq)}")
    print(f"OUTPUT task_type=SEMANTIC_SIMILARITY : nilai[0]={vs[0]:.6f}, dimensi={len(vs)}")

    selisih = np.linalg.norm(np.array(vd) - np.array(vq))
    print(f"\nSelisih (Euclidean distance) vektor DOCUMENT vs QUERY: {selisih:.6f}")
    print("Kesimpulan: task_type mengubah bobot representasi vektor walau input")
    print("teksnya identik -- ini relevan karena di Fase 1 kita pakai RETRIEVAL_DOCUMENT")
    print("dan di app.py kita pakai RETRIEVAL_QUERY untuk pasangan yang konsisten.")


def eksperimen_2_dimensionality():
    """
    Menguji trade-off ukuran vektor (output_dimensionality) terhadap
    kualitas retrieval -- bukan cuma ukuran file, tapi dampaknya ke skor
    similarity aktual pada database pasal yang ada.

    INPUT  : kueri uji yang sama, dengan 2 setting dimensi berbeda
    OUTPUT : Top-3 pasal hasil retrieval untuk tiap dimensi -- dibandingkan
             apakah hasilnya SAMA atau BERUBAH
    """
    cetak_header("EKSPERIMEN 2: Variasi output_dimensionality vs kualitas retrieval")
    print(f"INPUT (sama untuk keduanya): \"{KUERI_UJI}\"\n")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT pasal, vektor_embedding FROM dokumen_hukum")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[!] Database kosong, eksperimen ini butuh data dari fase1_indexing.py.")
        return

    daftar_pasal = [r[0] for r in rows]
    # Catatan: vektor di database tersimpan dalam dimensi default (3072).
    # Untuk uji dimensi lebih kecil secara adil, kita potong vektor database
    # ke N dimensi pertama sebagai pendekatan sederhana (bukan re-embed ulang).
    vektor_db_full = [np.array(json.loads(r[1])) for r in rows]

    for dim in [3072, 768, 256]:
        config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=dim)
        response = panggil_dengan_retry(
            client.models.embed_content,
            model="gemini-embedding-001", contents=KUERI_UJI, config=config
        )
        vektor_kueri = np.array(response.embeddings[0].values)

        vektor_db_dipotong = [v[:dim] for v in vektor_db_full]
        skor = [
            np.dot(vektor_kueri, v) / (np.linalg.norm(vektor_kueri) * np.linalg.norm(v))
            for v in vektor_db_dipotong
        ]
        top3_idx = np.argsort(skor)[-3:][::-1]
        top3 = [(daftar_pasal[i], round(float(skor[i]), 4)) for i in top3_idx]

        print(f"OUTPUT dim={dim:<5}: vektor kueri {len(vektor_kueri)} dimensi")
        print(f"           Top-3 hasil: {top3}\n")

    print("Kesimpulan: amati apakah Top-3 pasal yang muncul TETAP SAMA di ketiga")
    print("dimensi. Jika sama -> dimensi kecil cukup untuk dataset sekecil ini")
    print("(hemat storage tanpa kehilangan akurasi). Jika berbeda -> ada trade-off")
    print("kualitas yang harus dipertimbangkan sebelum memilih dimensi lebih kecil.")


def eksperimen_3_generation_config():
    """
    Menguji pengaruh parameter generation (temperature) terhadap konsistensi
    jawaban untuk pertanyaan yang sama, dijalankan 2x dengan temperature berbeda.

    INPUT  : prompt yang sama, dikirim 2x dengan temperature berbeda
    OUTPUT : dua jawaban -- dibandingkan variasi/konsistensinya
    """
    cetak_header("EKSPERIMEN 3: Pengaruh temperature pada konsistensi jawaban")
    prompt = "Ringkas dalam 2 kalimat: apa inti Pasal 27 UU ITE tentang muatan kesusilaan?"
    print(f"INPUT (sama untuk keduanya): \"{prompt}\"\n")

    for temp in [0.0, 1.0]:
        response = panggil_dengan_retry(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temp)
        )
        print(f"OUTPUT temperature={temp}:")
        print(f"  {response.text.strip()}\n")

    print("Kesimpulan: temperature=0.0 cenderung deterministik (cocok untuk")
    print("jawaban hukum yang butuh konsistensi), sedangkan temperature tinggi")
    print("lebih bervariasi -- relevan untuk dipilih di app.py (saat ini pakai 0.2).")


if __name__ == "__main__":
    print("EKSPLORASI FITUR GEMINI -- UNTUK REVISI DOSEN")
    eksperimen_1_task_type()
    eksperimen_2_dimensionality()
    eksperimen_3_generation_config()
    print("\n" + "=" * 70)
    print("[*] Eksplorasi selesai. Hasil di atas jadi dasar pembahasan laporan")
    print("    sebelum masuk ke perbandingan metode di fase3_evaluasi.py")
    print("=" * 70)