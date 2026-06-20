import os
import json
import sqlite3
import numpy as np
import time
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

# Standar SDK Gemini Terbaru
from google import genai
from google.genai import types

# 1. KONFIGURASI
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("API KEY =", GEMINI_API_KEY)

app = Flask(__name__)
client = genai.Client(api_key=GEMINI_API_KEY)
DB_NAME = "hukum_rag.db"

# ==========================================
# FUNGSI RETRIEVAL (STKI)
# ==========================================
def hitung_cosine_similarity(vec1, vec2):
    """Menghitung skor kemiripan antara dua vektor (kueri vs dokumen)"""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def get_top_k_dokumen(query_vector, k=3):
    """Mengambil Top-K chunk pasal paling relevan dari SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT pasal, teks_isi, vektor_embedding FROM dokumen_hukum")
    rows = cursor.fetchall()
    conn.close()

    hasil_pencarian = []
    for row in rows:
        pasal, teks_isi, vektor_json = row
        # Parse kembali string JSON dari SQLite menjadi Numpy Array
        vektor_dokumen = np.array(json.loads(vektor_json))
        
        # Kalkulasi manual kemiripan
        skor_sim = hitung_cosine_similarity(query_vector, vektor_dokumen)
        
        hasil_pencarian.append({
            "pasal": pasal,
            "teks_isi": teks_isi,
            "skor": float(skor_sim)
        })

    # Urutkan berdasarkan kemiripan tertinggi dan ambil sebatas nilai K
    hasil_pencarian = sorted(hasil_pencarian, key=lambda x: x["skor"], reverse=True)
    return hasil_pencarian[:k]

# ==========================================
# RUTE FLASK (ANTARMUKA)
# ==========================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tanya", methods=["POST"])
def tanya():
    data = request.json
    kueri_user = data.get("kueri")

    if not kueri_user:
        return jsonify({"error": "Kueri tidak boleh kosong"}), 400

    try:
        print("\n" + "="*50)
        print(f"[LOG] USER BERTANYA: {kueri_user}")
        
        # TAHAP 1: Embed Kueri User
        print("[TAHAP 1] Mengeksekusi Gemini Embedding API (Task: RETRIEVAL_QUERY)...")
        for i in range(5):
            try:
                response_embed = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=kueri_user,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY"
                    )
                )
                break

            except Exception as e:
                print(f"Percobaan {i+1} gagal")
                print(e)
                time.sleep(5)

        vektor_kueri = np.array(response_embed.embeddings[0].values)
        print(f"  -> Output: Vektor kueri sepanjang {len(vektor_kueri)} dimensi.")

        # TAHAP 2: Retrieval Dokumen Hukum
        print("[TAHAP 2] Mencari kemiripan Cosine Similarity dengan database SQLite...")
        top_dokumen = get_top_k_dokumen(vektor_kueri, k=3)
        print(f"  -> Output: Ditemukan {len(top_dokumen)} pasal relevan:")
        for d in top_dokumen:
            print(f"     - {d['pasal']} (Skor Kemiripan: {d['skor']:.4f})")

        # TAHAP 3: Penggabungan Konteks
        konteks_gabungan = "\n\n".join([f"Sumber: {doc['pasal']}\nIsi Regulasi: {doc['teks_isi']}" for doc in top_dokumen])

        # DEBUG: Lihat konteks yang dikirim ke Gemini
        print("\n===== KONTEKS YANG DIKIRIM KE GEMINI =====")
        print(konteks_gabungan)
        print("==========================================\n")

        # TAHAP 4: Generation
        print("[TAHAP 4] Mengeksekusi Gemini 2.5 Flash untuk generate jawaban...")
        prompt = f"""
        Anda adalah asisten hukum AI yang ramah dan objektif.

        Jawablah HANYA berdasarkan konteks regulasi yang diberikan.
        Jangan menggunakan pengetahuan di luar konteks.
        Jika informasi tidak ditemukan dalam konteks, katakan bahwa informasi tersebut tidak tersedia.

        Konteks Hukum:
        {konteks_gabungan}

        Pertanyaan:
        {kueri_user}
        """
        response_gen = None

        for i in range(5):
            try:
                response_gen = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2
                    )
                )
                break

            except Exception as e:
                print(f"Generate gagal percobaan {i+1}")
                print(e)
                time.sleep(5)

        if response_gen is None:
            return jsonify({
                "error": "Model Gemini 2.5 Flash sedang mengalami high demand. Silakan coba lagi beberapa saat."
            }), 503

        print(f"  -> Output Jawaban AI: {response_gen.text[:100]}... (dipotong)")
        print("="*50 + "\n")

        return jsonify({
            "jawaban": response_gen.text,
            "sumber": top_dokumen
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("[*] Menjalankan Server STKI Hukum Lokal...")
    app.run(debug=True, port=5000)