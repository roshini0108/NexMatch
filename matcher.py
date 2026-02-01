import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from feedback import load_feedback_scores
print("🔹 Loading model and index...")

model = SentenceTransformer("all-MiniLM-L6-v2")
data = pd.read_csv("data/suppliers.csv")
index = faiss.read_index("model/supplier.index")


# 🔹 Intent enrichment
def preprocess_query(query: str) -> str:
    q = query.lower()

    if "india" in q:
        q += " location india"
    if "high capacity" in q:
        q += " large scale manufacturing"
    if "automotive" in q:
        q += " automotive industry supplier"

    return q


# 🔹 Explainability
def explain_match(query: str, text: str) -> str:
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    overlap = q_words & t_words

    if overlap:
        return f"Matched on keywords: {', '.join(overlap)}"
    return "Semantic intent match based on supplier profile."


# 🔹 Auto supplier summary (🔥 THIS WAS MISSING AT RUNTIME)
def generate_summary(supplier):
    capacity = supplier.get("capacity", 0)

    if capacity >= 8000:
        cap_label = "high-capacity"
    elif capacity >= 3000:
        cap_label = "medium-capacity"
    else:
        cap_label = "small-scale"

    return (
        f"{supplier['company']} is a {cap_label} supplier based in "
        f"{supplier['location']} specializing in {supplier['product'].lower()}."
    )


# 🔥 MAIN MATCH FUNCTION (KEEP THIS LAST)
def find_matches(query: str, k: int = 5):
    feedback_scores = load_feedback_scores()
    query = preprocess_query(query)
    q_vec = model.encode([query]).astype("float32")

    distances, indices = index.search(q_vec, k)

    results = []

    for pos, idx in enumerate(indices[0]):
        supplier = data.iloc[int(idx)]
        dist = float(distances[0][pos])
        base_score = 1 / (1 + dist) * 100

        company_name = supplier["company"]
        fb = feedback_scores.get(company_name, {"positive": 0, "negative": 0})

        # 🔥 Feedback adjustment
        adjustment = (fb["positive"] * 3) - (fb["negative"] * 5)

        score = round(base_score + adjustment, 2)

        full_text = f"{supplier['product']} {supplier['description']}"

        results.append({
            "company": supplier["company"],
            "product": supplier["product"],
            "location": supplier["location"],
            "description": supplier["description"],
            "match_score": score,
            "summary": generate_summary(supplier),
            "reason": explain_match(query, full_text)
        })
        if score < 45:
            continue

    results = sorted(results, key=lambda x: x["match_score"], reverse=True)

    # 🔥 Cold-start fallback
    if not results or results[0]["match_score"] < 45:
        return [{
            "fallback": True,
            "message": "No exact supplier match found. Showing closest alternatives.",
            "alternatives": results[:3]
        }]

    return results