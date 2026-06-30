import os
import json
import sqlite3
import time
import numpy as np
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
DB_NAME = "hukum_rag.db"
K_RETRIEVAL = 3
MAX_RETRY = 5
JEDA_RETRY = 5

GROUND_TRUTH = [
    {"kueri": "Apa hukuman menyebarkan konten asusila secara online?", "pasal": ["Pasal 27", "Pasal 45"]},
    {"kueri": "Ringkas semua jenis pelanggaran yang diatur dalam Pasal 27 UU ITE", "pasal": ["Pasal 27"]},
    {"kueri": "Apa bedanya Pasal 27A dan Pasal 27B?", "pasal": ["Pasal 27A", "Pasal 27B"]},
    {"kueri": "Apakah mengancam orang lewat WhatsApp bisa dipidana?", "pasal": ["Pasal 29", "Pasal 45B"]},
    {"kueri": "Bandingkan hukuman untuk pencemaran nama baik vs pemerasan online", "pasal": ["Pasal 27A", "Pasal 27B", "Pasal 45"]},
    {"kueri": "Apakah penghinaan lewat TikTok termasuk Pasal 27A?", "pasal": ["Pasal 27A"]},
    {"kueri": "Kalau menghina lewat Facebook bagaimana hukumannya?", "pasal": ["Pasal 27A", "Pasal 45"]},
    {"kueri": "Apa itu sertifikat elektronik?", "pasal": ["DILUAR_CAKUPAN"]},
    {"kueri": "Bagaimana aturan tanda tangan elektronik?", "pasal": ["DILUAR_CAKUPAN"]}
]

def panggil_dengan_retry(fungsi_api, **kwargs):
    for percobaan in range(1, MAX_RETRY + 1):
        try:
            return fungsi_api(**kwargs)
        except Exception as e:
            print(f"    [RETRY {percobaan}/{MAX_RETRY}] Gagal: {e}")
            if percobaan < MAX_RETRY:
                time.sleep(JEDA_RETRY)
    raise RuntimeError(f"Gagal memanggil API Gemini setelah {MAX_RETRY} percobaan.")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_all_documents():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT pasal, teks_isi, vektor_embedding FROM dokumen_hukum")
    rows = cursor.fetchall()
    conn.close()
    return rows

def ekstrak_pasal_dari_teks(teks):
    matches = re.findall(r'Pasal\s*(\d+[A-Z]?)', teks, re.IGNORECASE)
    return set([f"Pasal {m.upper()}" for m in matches])

def jawab_tanpa_embedding(kueri):
    prompt = f"Jawab pertanyaan berikut mengenai UU ITE Indonesia.\n\nPertanyaan: {kueri}"
    
    start_waktu = time.perf_counter()
    response = panggil_dengan_retry(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt
    )
    durasi = time.perf_counter() - start_waktu
    return response.text, durasi

def jawab_dengan_embedding(kueri, dokumen_db):
    start_waktu = time.perf_counter()

    response_embed = panggil_dengan_retry(
        client.models.embed_content,
        model="gemini-embedding-001",
        contents=kueri,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    vektor_kueri = np.array(response_embed.embeddings[0].values)

    hasil = []
    for pasal, teks_isi, vektor_json in dokumen_db:
        vektor_doc = np.array(json.loads(vektor_json))
        skor = float(cosine_similarity(vektor_kueri, vektor_doc))
        hasil.append({"pasal": pasal, "teks_isi": teks_isi, "similarity": skor})

    hasil = sorted(hasil, key=lambda x: x["similarity"], reverse=True)[:K_RETRIEVAL]
    
    konteks = "\n\n".join([f"Sumber: {h['pasal']}\nIsi: {h['teks_isi']}" for h in hasil])
    jumlah_token_konteks = len(konteks.split())
    
    prompt = f"""Anda adalah asisten hukum Indonesia.
Gunakan HANYA informasi pada konteks.
Jangan menggunakan pengetahuan di luar konteks.
Jika informasi tidak tersedia di konteks, jawab:
"Informasi tersebut tidak ditemukan pada dokumen hukum yang tersedia."
Jika menyebut pasal, cantumkan nomor pasalnya.

Konteks Hukum:
{konteks}

Pertanyaan: {kueri}
Jawaban:"""

    response = panggil_dengan_retry(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt
    )
    durasi = time.perf_counter() - start_waktu
    
    sumber_retrieval = [{"pasal": h["pasal"], "similarity": h["similarity"]} for h in hasil]
    return response.text, sumber_retrieval, durasi, jumlah_token_konteks

if __name__ == "__main__":
    dokumen_db = get_all_documents()
    if not dokumen_db:
        print("[!] Database kosong. Jalankan fase indexing terlebih dahulu.")
        exit()

    log_lengkap = []
    stats = {
        "total": len(GROUND_TRUTH),
        "dir_benar": 0, "dir_salah": 0, "dir_hal": 0, "dir_time": 0.0,
        "rag_benar": 0, "rag_salah": 0, "rag_hal": 0, "rag_time": 0.0
    }

    print(f"[*] Menjalankan evaluasi untuk {stats['total']} query...\n")

    for item in GROUND_TRUTH:
        kueri = item["kueri"]
        target = set(item["pasal"])
        is_ood = "DILUAR_CAKUPAN" in target
        
        jwb_dir, time_dir = jawab_tanpa_embedding(kueri)
        jwb_rag, src_rag, time_rag, token_rag = jawab_dengan_embedding(kueri, dokumen_db)
        
        stats["dir_time"] += time_dir
        stats["rag_time"] += time_rag

        pasal_dir = ekstrak_pasal_dari_teks(jwb_dir)
        pasal_rag = ekstrak_pasal_dari_teks(jwb_rag)
        retrieved_rag = set([s["pasal"] for s in src_rag])

        dir_benar = False
        dir_halusinasi = False
        rag_benar = False
        rag_halusinasi = False

        if is_ood:
            if "tidak ditemukan" in jwb_rag.lower():
                rag_benar = True
            if len(pasal_rag) > 0:
                rag_halusinasi = True
                
            dir_benar = False
            if len(pasal_dir) > 0:
                dir_halusinasi = True
        else:
            if any(p in target for p in pasal_dir):
                dir_benar = True
            if any(p not in target for p in pasal_dir):
                dir_halusinasi = True
                
            if any(p in target for p in retrieved_rag) and any(p in target for p in pasal_rag):
                rag_benar = True
            if any(p not in target for p in pasal_rag):
                rag_halusinasi = True

        stats["dir_benar"] += 1 if dir_benar else 0
        stats["dir_salah"] += 1 if not dir_benar else 0
        stats["dir_hal"] += 1 if dir_halusinasi else 0
        
        stats["rag_benar"] += 1 if rag_benar else 0
        stats["rag_salah"] += 1 if not rag_benar else 0
        stats["rag_hal"] += 1 if rag_halusinasi else 0

        log_lengkap.append({
            "kueri": kueri,
            "ground_truth": list(target),
            "metode_direct": {
                "waktu_eksekusi_detik": round(time_dir, 3),
                "pasal_terdeteksi": list(pasal_dir),
                "is_correct": dir_benar,
                "is_hallucination": dir_halusinasi,
                "jawaban": jwb_dir
            },
            "metode_rag": {
                "waktu_eksekusi_detik": round(time_rag, 3),
                "jumlah_token_konteks": token_rag,
                "sumber_retrieval": src_rag,
                "pasal_terdeteksi": list(pasal_rag),
                "is_correct": rag_benar,
                "is_hallucination": rag_halusinasi,
                "jawaban": jwb_rag
            }
        })
        
        print(f"[+] Diproses: {kueri[:50]}...")

    with open("hasil_perbandingan_embedding.json", "w", encoding="utf-8") as f:
        json.dump(log_lengkap, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 51)
    print("HASIL PERBANDINGAN")
    print("=" * 51)
    print(f"{'Jumlah Query':<25} : {stats['total']}")
    print("-" * 51)
    print(f"{'Direct benar':<25} : {stats['dir_benar']}")
    print(f"{'RAG benar':<25} : {stats['rag_benar']}")
    print(f"{'Direct salah':<25} : {stats['dir_salah']}")
    print(f"{'RAG salah':<25} : {stats['rag_salah']}")
    print("-" * 51)
    print(f"{'Direct halusinasi':<25} : {stats['dir_hal']}")
    print(f"{'RAG halusinasi':<25} : {stats['rag_hal']}")
    print("-" * 51)
    print(f"{'Rata-rata waktu Direct':<25} : {(stats['dir_time']/stats['total']):.3f} detik")
    print(f"{'Rata-rata waktu RAG':<25} : {(stats['rag_time']/stats['total']):.3f} detik")
    print("=" * 51)
    print("[*] Log detail tersimpan di: hasil_perbandingan_embedding.json")