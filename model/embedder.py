from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

print("Loading AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Reading supplier data...")
data = pd.read_csv("data/suppliers.csv")

texts = data["description"].tolist()

print("Generating embeddings...")
embeddings = model.encode(texts)

print("Building vector index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "model/supplier.index")

print("NexMatch embedding engine ready.")
