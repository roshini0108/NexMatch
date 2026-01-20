import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

# Load dataset
data_path = "data/suppliers.csv"

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found! Please create the data folder and CSV file.")
else:
    data = pd.read_csv(data_path)
    data.fillna("", inplace=True)

    # Combine columns for semantic context
    data['combined_text'] = data['product'] + " " + data['category'] + " " + data['description']

    # Load AI model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    print("Generating AI embeddings for 30 rows...")
    embeddings = model.encode(data['combined_text'].tolist())
    vectors = np.array(embeddings).astype('float32')

    # Create and save FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Ensure model folder exists and save index
    os.makedirs("model", exist_ok=True)
    faiss.write_index(index, "model/company.index")
    print("✅ Success: model/company.index has been created.")