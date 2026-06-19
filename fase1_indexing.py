import os
import json
import re
import sqlite3
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan. Pastikan file .env sudah dikonfigurasi.")

client = genai.Client(api_key=GEMINI_API_KEY)
DB_NAME = "hukum_rag.db"

def setup_database():
    """Membuat database dan tabel SQLite jika belum ada."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dokumen_hukum (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama_dokumen TEXT,
            pasal TEXT,
            teks_isi TEXT,
            vektor_embedding TEXT
        )
    ''')
    conn.commit()
    return conn

def ekstrak_dan_bersihkan_pdf(pdf_path):
    """Mengekstrak teks dari PDF dan membersihkan noise administratif."""
    teks_gabungan = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            teks_halaman = page.extract_text()
            if not teks_halaman:
                continue
                
            baris_teks = teks_halaman.split('\n')
            baris_bersih = []
            
            for baris in baris_teks:
                baris = baris.strip()
                # Filter noise lembaran negara resmi
                is_noise = (
                    "PRESIDEN" in baris or 
                    "REPUBLIK INDONESIA" in baris or 
                    "SK No" in baris or 
                    re.match(r'^-\s*\d+\s*-$', baris) # Menangkap format halaman "- 7 -" atau "-7-"
                )
                
                if is_noise or not baris:
                    continue
                    
                baris_bersih.append(baris)
            
            teks_gabungan += " ".join(baris_bersih) + " "
            
    teks_gabungan = re.sub(r'\s+', ' ', teks_gabungan).strip()
    return teks_gabungan

def chunking_per_pasal(teks_bersih):
    """Memecah teks raksasa menjadi potongan semantik berdasarkan kata 'Pasal'."""
    # Pola RegEx untuk mencari "Pasal [Angka][Huruf Opsional]" (Contoh: Pasal 27, Pasal 27A)
    pola_pasal = r'(Pasal \d+[A-Z]?)'
    
    # re.split akan memecah teks dan menyimpan delimiter (kata pasalnya)
    pecahan = re.split(pola_pasal, teks_bersih)
    chunks = []
    
    # Pecahan berformat: [teks_awal, "Pasal 27", "isi pasal 27...", "Pasal 28", "isi pasal 28..."]
    for i in range(1, len(pecahan), 2):
        nama_pasal = pecahan[i].strip()
        isi_pasal = pecahan[i+1].strip()
        
        teks_utuh = f"{nama_pasal} {isi_pasal}"
        chunks.append({
            "pasal": nama_pasal,
            "teks": teks_utuh
        })
        
    return chunks

def proses_embedding_dan_simpan(chunks, conn, nama_dokumen):
    """Mengubah chunk menjadi embedding dan menyimpannya ke SQLite."""
    cursor = conn.cursor()

    # Fokus topik tertentu sesuai revisi dosen
    topik_fokus = ["Pasal 27", "Pasal 27A", "Pasal 27B", "Pasal 28", "Pasal 29"]
    chunks_terfilter = [
        c for c in chunks
        if any(c["pasal"].startswith(p) for p in topik_fokus)
    ]

    print(f"\n[*] Memproses {len(chunks_terfilter)} chunk khusus Pasal 27-29")

    for i, chunk in enumerate(chunks_terfilter, 1):

        print("\n" + "=" * 50)
        print(f"[{i}/{len(chunks_terfilter)}] {chunk['pasal']}")

        teks = chunk["teks"]

        print("\n[INPUT TEKS]")
        print(teks[:500])
        print()

        # ===== FILTER CHUNK BURUK =====
        if len(teks.strip()) == 0:
            print("Chunk kosong, dilewati.")
            continue

        if len(teks) < 100:
            print("Chunk terlalu pendek, dilewati.")
            continue

        if "diubah sehingga berbunyi" in teks:
            print("Chunk referensi perubahan UU, dilewati.")
            continue

        if teks.endswith("dan"):
            print("Chunk tidak lengkap, dilewati.")
            continue

        if "Pasal." in teks and len(teks) < 200:
            print("Chunk referensi pasal, dilewati.")
            continue
        # =============================

        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=teks,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )

            vektor = response.embeddings[0].values

            print("[OUTPUT VEKTOR]")
            print(f"Dimensi embedding = {len(vektor)}")

            cursor.execute(
                '''
                INSERT INTO dokumen_hukum
                (nama_dokumen, pasal, teks_isi, vektor_embedding)
                VALUES (?, ?, ?, ?)
                ''',
                (
                    nama_dokumen,
                    chunk["pasal"],
                    teks,
                    json.dumps(vektor)
                )
            )

            conn.commit()

        except Exception as e:
            print(f"ERROR pada {chunk['pasal']}")
            print(e)

    print("\n[*] Penyimpanan ke SQLite selesai.")

if __name__ == "__main__":
    file_pdf = "dataset/uu_ite.pdf"
    
    if not os.path.exists(file_pdf):
        print(f"File {file_pdf} tidak ditemukan! Pastikan file sudah ada di folder 'dataset'.")
        exit()

    print("\n--- MULAI FASE 1: INDEXING ---")
    
    print("\n[1] Membuka koneksi database SQLite...")
    db_koneksi = setup_database()
    
    print("\n[2] Mengekstrak teks dari PDF dan membersihkan noise administratif...")
    teks_mentah = ekstrak_dan_bersihkan_pdf(file_pdf)
    
    print("\n[3] Memecah teks menggunakan metode Semantic Chunking...")
    daftar_chunk = chunking_per_pasal(teks_mentah)
    
    if daftar_chunk:
        print("\n[4] Mengubah teks menjadi Embedding Vektor dan menyimpan ke Database...")
        proses_embedding_dan_simpan(daftar_chunk, db_koneksi, "UU ITE")
    else:
        print("\n[!] Gagal membuat chunk. Periksa kembali struktur teks PDF Anda.")
        
    db_koneksi.close()
    print("\n--- FASE 1 SELESAI ---")