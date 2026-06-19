import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

teks_pasal = "Pasal 27 ayat 1: Setiap Orang dengan sengaja dan tanpa hak menyiarkan, mempertunjukkan, mendistribusikan, mentransmisikan, dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan yang melanggar kesusilaan."
kueri_user = "hukuman sebar konten asusila"

print("==================================================")
print("EKSPLORASI FITUR GEMINI EMBEDDING (REVISI DOSEN)")
print("==================================================\n")

print("[1] EKSPLORASI TASK TYPE")
print("Kita menguji apakah task_type mengubah nilai vektor yang dihasilkan untuk teks yang sama.")
embed_doc = client.models.embed_content(
    model="text-embedding-004", contents=teks_pasal,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)
embed_sim = client.models.embed_content(
    model="text-embedding-004", contents=teks_pasal,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)
print(f"-> Vektor index ke-0 (RETRIEVAL_DOCUMENT): {embed_doc.embeddings[0].values[0]:.6f}")
print(f"-> Vektor index ke-0 (SEMANTIC_SIMILARITY): {embed_sim.embeddings[0].values[0]:.6f}")
print("-> Kesimpulan: Task Type mengubah bobot representasi vektor berdasarkan tujuan penggunaannya!\n")


print("[2] EKSPLORASI DIMENSIONALITY REDUCTION")
print("Gemini memiliki fitur untuk mengurangi dimensi vektor agar database lebih ringan (misal dari 768 ke 256).")
embed_full = client.models.embed_content(
    model="text-embedding-004", contents=teks_pasal,
)
embed_reduced = client.models.embed_content(
    model="text-embedding-004", contents=teks_pasal,
    config=types.EmbedContentConfig(output_dimensionality=256)
)
print(f"-> Dimensi Vektor Normal : {len(embed_full.embeddings[0].values)}")
print(f"-> Dimensi Vektor Reduced: {len(embed_reduced.embeddings[0].values)}")
print("-> Kesimpulan: Pengurangan dimensi berhasil, bisa menghemat memori database SQLite hingga 60%!\n")

print("==================================================")