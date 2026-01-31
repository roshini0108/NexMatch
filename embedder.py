import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Reading suppliers.csv...")
data = pd.read_csv("data/suppliers.csv")

texts = data["description"].astype(str).tolist()

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True
).astype("float32")

print("Creating FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

os.makedirs("model", exist_ok=True)
faiss.write_index(index, "model/supplier.index")

print("✅ supplier.index saved inside /model")