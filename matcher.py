import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

print("Loading model + index...")

model = SentenceTransformer("all-MiniLM-L6-v2")
data = pd.read_csv("data/suppliers.csv")
index = faiss.read_index("model/supplier.index")


def find_matches(query, k=3):
    q_vec = model.encode([query]).astype("float32")

    distances, indices = index.search(q_vec, k)

    results = []

    for pos, idx in enumerate(indices[0]):
        supplier = data.iloc[int(idx)]

        dist = float(distances[0][pos])

        # smooth scoring
        score = round(1 / (1 + dist) * 100, 2)

        results.append({
            "company": supplier["company"],
            "product": supplier["product"],
            "location": supplier["location"],
            "description": supplier["description"],
            "match_score": score,
            "reason": f"Semantic similarity match for '{query}'."
        })

    return sorted(results, key=lambda x: x["match_score"], reverse=True)