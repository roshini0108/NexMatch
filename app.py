from flask import Flask, request, jsonify, render_template
from matcher import find_matches
from feedback import save_feedback

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    results = find_matches(query)
    return jsonify(results)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "engine": "SentenceTransformers + FAISS",
        "service": "NexMatch AI"
    })
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()

    query = data.get("query")
    company = data.get("company")
    feedback = data.get("feedback")  # "positive" or "negative"

    if not all([query, company, feedback]):
        return jsonify({"error": "Invalid feedback data"}), 400

    save_feedback(query, company, feedback)
    return jsonify({"status": "feedback saved"})

if __name__ == "__main__":
    app.run(debug=True)