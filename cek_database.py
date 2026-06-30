"""
CEK ISI DATABASE -- Diagnostik Cepat
=======================================
Jalankan ini untuk melihat pasal apa saja yang BERHASIL ter-index di
hukum_rag.db, supaya bisa dicocokkan dengan ground truth di fase3_evaluasi.py.

Kalau ada pasal di ground truth yang TIDAK muncul di sini, itu sumber
kegagalan retrieval -- bukan masalah kualitas embedding, tapi masalah
data yang memang tidak ada untuk dicari.
"""

import sqlite3
import json

DB_NAME = "hukum_rag.db"

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("SELECT pasal, teks_isi FROM dokumen_hukum ORDER BY pasal")
rows = cursor.fetchall()
conn.close()

print(f"[*] Total pasal ter-index di database: {len(rows)}\n")
print("=" * 70)
print("DAFTAR PASAL YANG ADA DI DATABASE")
print("=" * 70)
for pasal, teks in rows:
    print(f"  {pasal:<15} ({len(teks)} karakter) : \"{teks[:60]}...\"")

# Cek khusus pasal yang dicurigai hilang berdasarkan hasil evaluasi
print("\n" + "=" * 70)
print("CEK KHUSUS -- Pasal yang dicurigai hilang dari hasil evaluasi")
print("=" * 70)

pasal_dicek = ["Pasal 29", "Pasal 27B", "Pasal 45", "Pasal 45A", "Pasal 45B"]
pasal_ada = [p for p, _ in rows]

for p in pasal_dicek:
    status = "ADA" if p in pasal_ada else "TIDAK ADA / HILANG"
    print(f"  {p:<15} -> {status}")

print("\n[*] Jika ada yang 'TIDAK ADA / HILANG', berarti filter di fase1_indexing.py")
print("    salah membuang chunk pasal tersebut, atau chunking gagal menangkapnya.")
print("    Solusi: jalankan ulang fase1_indexing.py dengan mode debug (lihat saran")
print("    di chat) untuk melihat KENAPA chunk pasal itu di-skip.")
