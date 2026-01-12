from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("model/supplier.index")
data = pd.read_csv("data/suppliers.csv")

def find_matches(query, k=3):
    q_vector = model.encode([query])
    distances, indices = index.search(np.array(q_vector), k)

    results = []

    for pos, i in enumerate(indices[0]):
        supplier = data.iloc[i]

        # 🔢 Base score from semantic similarity
        score = round(100 - distances[0][pos] * 10, 2)

        # 🏭 PART 3 — Business Rules (THIS IS WHERE IT GOES)
        if "india" in query.lower() and supplier["location"].lower() == "india":
            score += 5

        # 🧾 Explainability
        reason = f"Matched on product relevance and industry alignment. Location: {supplier['location']}."

        results.append({
            "company": supplier["company"],
            "product": supplier["product"],
            "location": supplier["location"],
            "description": supplier["description"],
            "match_score": score,
            "reason": reason
        })

    # 🥇 PART 2 — Ranking (AFTER the loop, before return)
    results = sorted(results, key=lambda x: x["match_score"], reverse=True)

    return results
