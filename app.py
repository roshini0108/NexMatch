from flask import Flask, request, jsonify, render_template
from matcher import find_matches

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/match", methods=["POST"])
def match():
    query = request.json.get("query", "")
    results = find_matches(query)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)