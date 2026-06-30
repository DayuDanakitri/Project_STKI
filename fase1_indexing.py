import os
import json
import re
import sqlite3
import pdfplumber
import hashlib
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from teks_override_manual import TEKS_OVERRIDE_MANUAL

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan. Pastikan file .env sudah dikonfigurasi.")

client = genai.Client(api_key=GEMINI_API_KEY)
DB_NAME = "hukum_rag.db"

# 7. Menggunakan Set untuk lookup O(1) yang lebih cepat
TOPIK_FOKUS = {"Pasal 27", "Pasal 27A", "Pasal 27B", "Pasal 28", "Pasal 29",
               "Pasal 45", "Pasal 45A", "Pasal 45B"}

def setup_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dokumen_hukum(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nama_dokumen TEXT,
        pasal TEXT,
        teks_isi TEXT,
        vektor_embedding TEXT,
        hash_teks TEXT UNIQUE
    )
    """)
    conn.commit()
    return conn

def hash_chunk(teks):
    return hashlib.sha256(teks.encode("utf-8")).hexdigest()

def reset_data_dokumen(conn, nama_dokumen):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dokumen_hukum WHERE nama_dokumen = ?", (nama_dokumen,))
    conn.commit()
    print(f"[RESET] Data lama untuk '{nama_dokumen}' telah dihapus.")

def ekstrak_dan_bersihkan_pdf(pdf_path):
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
                is_noise = (
                    "PRESIDEN" in baris or
                    "REPUBLIK INDONESIA" in baris or
                    "SK No" in baris or
                    re.match(r'^-\s*\d+\s*-$', baris)
                )
                if is_noise or not baris:
                    continue
                baris_bersih.append(baris)
            teks_gabungan += " ".join(baris_bersih) + " "
    return re.sub(r'\s+', ' ', teks_gabungan).strip()

def normalisasi_label_pasal(teks):
    perbaikan = [
        (r'Pasal\s*278\b', 'Pasal 27B'),
        (r'Pasal\s*2TA\b', 'Pasal 27A'),
        (r'Pasil', 'Pasal'),
        (r'PasaI', 'Pasal'),
        (r'ha1', 'hal'),
        (r'Rp4O0', 'Rp400'),
        (r'Rp75O', 'Rp750'),
        (r'0rang', 'Orang'),
        (r'Iima', 'lima'),
        (r'perbuatan \.\.\.', 'Perbuatan'),
    ]
    for pola, hasil in perbaikan:
        teks = re.sub(pola, hasil, teks, flags=re.IGNORECASE)
    return teks

def chunking_berbasis_jangkar(teks_bersih):
    pola_jangkar = (
        r'(?:Ketentuan (Pasal\s*\d+[A-Z]?)[^.]*?sehingga berbunyi sebagai berikut:'
        r'|disisipkan[^.]*?sehingga berbunyi sebagai berikut:)'
    )
    anchors = list(re.finditer(pola_jangkar, teks_bersih))
    anchors.sort(key=lambda m: m.start())

    chunks_mentah = []
    for i, m in enumerate(anchors):
        start_isi = m.end()
        end_isi = anchors[i + 1].start() if i + 1 < len(anchors) else len(teks_bersih)
        chunks_mentah.append(teks_bersih[start_isi:end_isi].strip())

    pola_label = r'\bPasal\s+(\d+[A-Z]?)\b'
    chunks_final = {}

    for cm in chunks_mentah:
        label_matches = list(re.finditer(pola_label, cm))
        for j, lm in enumerate(label_matches):
            nama = f"Pasal {lm.group(1)}"
            konteks_sebelum = cm[max(0, lm.start() - 30):lm.start()]
            
            if j > 0 and any(k in konteks_sebelum for k in ["dimaksud dalam", "antara", "yakni"]):
                continue

            start = lm.start()
            end = label_matches[j + 1].start() if j + 1 < len(label_matches) else len(cm)
            potongan = cm[start:end].strip()

            if len(potongan) < 30:
                continue

            if nama not in chunks_final or len(potongan) > len(chunks_final[nama]):
                chunks_final[nama] = potongan

    return [{"pasal": nama, "teks": teks} for nama, teks in chunks_final.items()]

def terapkan_override_manual(chunks, override_dict):
    chunks_dict = {c["pasal"]: c["teks"] for c in chunks}
    for nama_pasal, teks_lengkap in override_dict.items():
        # 8. Log eksplisit untuk Override manual
        print(f"[OVERRIDE] Override digunakan untuk {nama_pasal}.")
        chunks_dict[nama_pasal] = teks_lengkap
    return [{"pasal": nama, "teks": teks} for nama, teks in chunks_dict.items()]

def proses_embedding_dan_simpan(chunks, conn, nama_dokumen):
    cursor = conn.cursor()
    chunks_topik = [c for c in chunks if c["pasal"] in TOPIK_FOKUS]

    for chunk in chunks_topik:
        teks = chunk["teks"].strip()
        
        # 2. Filter panjang karakter lebih toleran
        if len(teks) < 80 or len(teks.split()) < 20:
            print(f"[-] {chunk['pasal']} dilewati -> terlalu pendek.")
            continue
            
        # 3. Filter teks terpotong lebih aman
        if teks.endswith(("dan", "atau", ",")) or "..." in teks:
            print(f"[-] {chunk['pasal']} dilewati -> terindikasi terpotong.")
            continue
            
        if "sehingga berbunyi" in teks:
            print(f"[-] {chunk['pasal']} dilewati -> masih berupa teks pengantar.")
            continue

        # 6. Logging progres yang baik untuk presentasi
        print(f"[*] Embedding {chunk['pasal']} ... ({len(teks)} karakter)")
        
        vektor = None
        max_retries = 5
        
        # 4 & 5. Try/Except block dan mekanisme Retry
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=teks,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                vektor = response.embeddings[0].values
                break
            except Exception as e:
                print(f"    [!] Error API (percobaan {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        
        if not vektor:
            print(f"    [!] Gagal embed {chunk['pasal']} setelah {max_retries} percobaan.")
            continue
            
        print(f"    [+] Selesai! ({len(vektor)} dimensi)")

        cursor.execute(
            """
            INSERT OR IGNORE INTO dokumen_hukum
            (nama_dokumen,pasal,teks_isi,vektor_embedding,hash_teks)
            VALUES (?,?,?,?,?)
            """,
            (nama_dokumen, chunk["pasal"], teks, json.dumps(vektor), hash_chunk(teks))
        )
    conn.commit()

def verifikasi_tidak_ada_duplikat(conn, nama_dokumen):
    cursor = conn.cursor()
    # 1. Mengecek duplikat murni dari konten isinya (hash), bukan cuma nama pasalnya
    cursor.execute('''
        SELECT hash_teks, COUNT(*) as jumlah
        FROM dokumen_hukum
        WHERE nama_dokumen = ?
        GROUP BY hash_teks
        HAVING jumlah > 1
    ''', (nama_dokumen,))
    
    duplikat = cursor.fetchall()
    if duplikat:
        raise ValueError(f"Ditemukan duplikat teks di database berdasarkan hash: {duplikat}")
    else:
        print("[OK] Verifikasi aman: Tidak ada duplikat berdasarkan hash.")

if __name__ == "__main__":
    file_pdf = "dataset/uu_ite.pdf"
    
    if not os.path.exists(file_pdf):
        raise FileNotFoundError(f"File {file_pdf} tidak ditemukan di folder 'dataset'.")

    print("--- MEMULAI PROSES INDEXING ---")
    nama_dokumen = "UU ITE"
    db_koneksi = setup_database()
    
    reset_data_dokumen(db_koneksi, nama_dokumen)
    
    print("[*] Mengekstrak teks dari PDF...")
    teks_mentah = ekstrak_dan_bersihkan_pdf(file_pdf)
    teks_mentah = normalisasi_label_pasal(teks_mentah)
    
    print("[*] Melakukan chunking...")
    daftar_chunk = chunking_berbasis_jangkar(teks_mentah)
    daftar_chunk = terapkan_override_manual(daftar_chunk, TEKS_OVERRIDE_MANUAL)

    if daftar_chunk:
        print("\n--- MEMULAI PROSES EMBEDDING ---")
        proses_embedding_dan_simpan(daftar_chunk, db_koneksi, nama_dokumen)
        
        print("\n--- MELAKUKAN VERIFIKASI ---")
        verifikasi_tidak_ada_duplikat(db_koneksi, nama_dokumen)
        print("\n[+] Proses indexing selesai dengan sukses.")
    else:
        print("Gagal membuat chunk. Periksa kembali struktur teks PDF Anda.")
        
    db_koneksi.close()
    print("\n--- FASE 1 SELESAI ---")
