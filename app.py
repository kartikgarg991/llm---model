from flask import Flask, render_template, request, jsonify
from backend import get_answer  # your RAG / AI logic

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "")
    answer = get_answer(query)  # your function that returns AI response
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
