import sqlite3
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
DB_NAME = "hukum_rag.db"

# ambil dokumen SQLite
def get_all_documents():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT pasal, teks_isi, vektor_embedding FROM dokumen_hukum")
    rows = cursor.fetchall()
    conn.close()
    return rows

dokumen_db = get_all_documents()
daftar_pasal = [row[0] for row in dokumen_db]
corpus_teks = [row[1] for row in dokumen_db]
vektor_gemini_db = [np.array(json.loads(row[2])) for row in dokumen_db]


print("[*] Melatih model TF-IDF...")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus_teks)

##dataset gound truth
ground_truth = {
    "hukum menyebarkan berita bohong atau hoaks yang merugikan konsumen": ["Pasal 28", "Pasal 45A"],
    "hukuman menyebar konten asusila atau pornografi": ["Pasal 27", "Pasal 45"],
    "mengancam orang lain secara elektronik": ["Pasal 29", "Pasal 45B"],
    "memeras orang dengan ancaman menyebarkan rahasia": ["Pasal 27B"]
}

#evaluasi
def hitung_metrik(retrieved_docs, relevant_docs, k):
    """Menghitung Precision@k dan Recall@k"""
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    # Irisan (Intersection) antara yang ditarik dan yang relevan
    true_positives = len(retrieved_set.intersection(relevant_set))
    
    precision = true_positives / k
    recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    return precision, recall

#kcompare
K_PENCARIAN = 3
hasil_evaluasi = []

print(f"[*] Memulai Evaluasi Komparatif (Top-{K_PENCARIAN})...\n")
print("-" * 60)

for kueri, jawaban_benar in ground_truth.items():
    print(f"Kueri: '{kueri}'")
    print(f"Ground Truth (Harapan): {jawaban_benar}")
    
    query_vec_tfidf = vectorizer.transform([kueri])
    sim_scores_tfidf = cosine_similarity(query_vec_tfidf, tfidf_matrix).flatten()
    
    top_indices_tfidf = sim_scores_tfidf.argsort()[-K_PENCARIAN:][::-1]
    retrieved_tfidf = [daftar_pasal[i] for i in top_indices_tfidf]
    
    p_tfidf, r_tfidf = hitung_metrik(retrieved_tfidf, jawaban_benar, K_PENCARIAN)
    
    response_embed = client.models.embed_content(
        model="gemini-embedding-001",
        contents=kueri,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_vec_gemini = np.array(response_embed.embeddings[0].values)
    
    # Hitung cosine similarity
    sim_scores_gemini = [
        np.dot(query_vec_gemini, doc_vec) / (np.linalg.norm(query_vec_gemini) * np.linalg.norm(doc_vec))
        for doc_vec in vektor_gemini_db
    ]
    
    # Ambil index dengan skor tertinggi
    top_indices_gemini = np.argsort(sim_scores_gemini)[-K_PENCARIAN:][::-1]
    retrieved_gemini = [daftar_pasal[i] for i in top_indices_gemini]
    
    p_gemini, r_gemini = hitung_metrik(retrieved_gemini, jawaban_benar, K_PENCARIAN)
    
    # cetak hasil/query
    print(f"  [TF-IDF] Menarik: {retrieved_tfidf} | Precision: {p_tfidf:.2f}, Recall: {r_tfidf:.2f}")
    print(f"  [GEMINI] Menarik: {retrieved_gemini} | Precision: {p_gemini:.2f}, Recall: {r_gemini:.2f}\n")
    
    hasil_evaluasi.append({
        "p_tfidf": p_tfidf, "r_tfidf": r_tfidf,
        "p_gemini": p_gemini, "r_gemini": r_gemini
    })

#rekap rata2
avg_p_tfidf = sum(x["p_tfidf"] for x in hasil_evaluasi) / len(hasil_evaluasi)
avg_r_tfidf = sum(x["r_tfidf"] for x in hasil_evaluasi) / len(hasil_evaluasi)
avg_p_gemini = sum(x["p_gemini"] for x in hasil_evaluasi) / len(hasil_evaluasi)
avg_r_gemini = sum(x["r_gemini"] for x in hasil_evaluasi) / len(hasil_evaluasi)

print("=" * 60)
print("KESIMPULAN EVALUASI (RATA-RATA)")
print("=" * 60)
print(f"Metode TF-IDF Baseline   -> Precision@{K_PENCARIAN}: {avg_p_tfidf:.2f} | Recall@{K_PENCARIAN}: {avg_r_tfidf:.2f}")
print(f"Metode Gemini Embedding  -> Precision@{K_PENCARIAN}: {avg_p_gemini:.2f} | Recall@{K_PENCARIAN}: {avg_r_gemini:.2f}")