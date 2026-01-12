from flask import Flask, request, jsonify, render_template
from model.matcher import find_matches

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/match", methods=["POST"])
def match():
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({"error": "Missing query"}), 400

        query = data["query"]
        results = find_matches(query)

        return jsonify(results)

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
