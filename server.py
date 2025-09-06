from dotenv import load_dotenv
load_dotenv()


from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from index import index_document
from query import chat, embeddings, pinecone_index, rewrite_query, conversation_history




app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve frontend
@app.route("/")
def home():
    return render_template("index.html")

# Health check
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# Upload PDF
@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({"error": "No file"}), 400

        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"Saved file to {filepath}")

        # Index PDF in Pinecone
        index_document(filepath)
        print("Indexing complete")
        
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        print("Upload error:", e)  # <--- This will show the error in your terminal
        import traceback
        traceback.print_exc()      # <--- This will print the full traceback
        return jsonify({"error": str(e)}), 500   

# Ask query
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query"}), 400

    # Rewrite query
    prev_conv = "\n".join(conversation_history[-3:])
    std_query = rewrite_query(prev_conv, query)

    # Get vector & query Pinecone
    query_vector = embeddings.embed_query(std_query)
    raw_results = pinecone_index.query(vector=query_vector, top_k=10, include_metadata=True)
    matches = raw_results["matches"]

    if not matches:
        answer = "No relevant content found in the document."
    else:
        context = "\n\n---\n\n".join([m["metadata"]["text"] for m in matches])
        prompt = f"""
        You are an instructor who answers only from the given context.
        Previous Conversation (last 3 turns):
        {prev_conv}
        User Query: {std_query}
        Context:
        {context}
        Answer strictly using ONLY the context above. If not relevant, say 'No relevant content found in the document.'
        """
        answer = chat.send_message(prompt).text

    # Update conversation history
    conversation_history.append(f"User: {query}\nAssistant: {answer}")
    conversation_history[:] = conversation_history[-3:]

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
