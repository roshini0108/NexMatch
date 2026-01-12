import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load model and data once
model = SentenceTransformer("all-MiniLM-L6-v2")
data = pd.read_csv("data/suppliers.csv")

# Load FAISS index
index = faiss.read_index("model/supplier.index")

def find_matches(query, k=5):
    print("Understanding buyer intent...")
    q_vector = model.encode([query])
    distances, indices = index.search(np.array(q_vector), k)

    results = []

    for pos, i in enumerate(indices[0]):
        supplier = data.iloc[i]

        # Convert numpy float -> Python float
        score = float(round(100 - distances[0][pos] * 10, 2))

        reason = f"Matched based on product relevance and location: {supplier['location']}"

        results.append({
            "company": supplier["company"],
            "product": supplier["product"],
            "location": supplier["location"],
            "description": supplier["description"],
            "match_score": score,
            "reason": reason
        })

    return results
