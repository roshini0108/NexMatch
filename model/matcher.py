from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("model/supplier.index")
data = pd.read_csv("data/suppliers.csv")

def find_matches(query, k=3):
    print("Understanding buyer intent...")
    q_vector = model.encode([query])

    print("Searching best suppliers...")
    distances, indices = index.search(np.array(q_vector), k)

    results = []
    for i in indices[0]:
        supplier = data.iloc[i]
        results.append({
            "company": supplier["company"],
            "product": supplier["product"],
            "location": supplier["location"],
            "description": supplier["description"]
        })

    return results
