from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import traceback
import tempfile

from index import index_document
from query import chat, embeddings, pinecone_index, rewrite_query, conversation_history

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        file_bytes = file.read()  # ✅ Memory mein lo

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                temp_file_path = tmp.name

            index_document(temp_file_path)  # ✅ Pinecone mein index

            return jsonify({"success": True, "filename": filename})

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)  # ✅ Delete temp file

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query"}), 400

    prev_conv = "\n".join(conversation_history[-5:])
    std_query = rewrite_query(prev_conv, query)

    query_vector = embeddings.embed_query(std_query)
    raw_results = pinecone_index.query(vector=query_vector, top_k=10, include_metadata=True)
    matches = raw_results["matches"]

    if not matches:
        answer = "Sorry, I couldn't find anything in the document for that query."
    else:
        context = "\n\n---\n\n".join([m["metadata"]["text"] for m in matches])
        prompt = f"""
        You are an instructor who answers only from the given context.
        
        Previous Conversation:
        {prev_conv}

        User Query: {std_query}

        Context: {context}

        Answer strictly using ONLY the context above.
        """
        answer = chat.send_message(prompt).text

    conversation_history.append(f"User: {query}\nAssistant: {answer}")
    conversation_history[:] = conversation_history[-5:]

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    
