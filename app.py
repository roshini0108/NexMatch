from flask import Flask, request, jsonify
from model.matcher import find_matches

app = Flask(__name__)

@app.route("/")
def home():
    return "NexMatch AI Engine is running."

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    query = data.get("query")

    results = find_matches(query)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
