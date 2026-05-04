from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import traceback
import tempfile
import time
import threading 
from index import index_document
from query import chat, embeddings, pinecone_index, rewrite_query, conversation_history , search_context

session_registry = {}
SESSION_TTL_HOURS = 2

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    session_id = request.form.get("session_id")
    if not session_id:
        return jsonify({"error": "No session_id"}), 400

    session_registry[session_id] = time.time() 


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
            ext = os.path.splitext(secure_filename(file.filename))[1]  # gets .docx, .txt etc
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_bytes)
                temp_file_path = tmp.name

            index_document(temp_file_path, session_id)  # ✅ Pinecone mein index

            return jsonify({"success": True, "filename": filename})

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)  # ✅ Delete temp file

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "No session_id"}), 400
            
        session_registry[session_id] = time.time()
        query = data.get("query")
        if not query:
            return jsonify({"error": "No query"}), 400

        prev_conv = "\n".join(conversation_history[-5:])
        std_query = rewrite_query(prev_conv, query)

        context = search_context(session_id, std_query)
        
        if not context:
            answer = "Sorry, I couldn't find anything in the document for that query."
        else:
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

    except Exception as e:
        traceback.print_exc()
        error_text = str(e)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
            return jsonify({"error": "Gemini API limit exhausted. Please try again after some time."}), 429
        return jsonify({"error": error_text}), 500


def cleanup_old_namespaces():
    from pinecone import Pinecone # Import inside thread to avoid issues
    while True:
        time.sleep(3600)  # check every hour
        now = time.time()
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        
        expired = [
            sid for sid, last_active in session_registry.items()
            if now - last_active > SESSION_TTL_HOURS * 3600
        ]
        for sid in expired:
            try:
                index.delete(delete_all=True, namespace=sid)
                del session_registry[sid]
                print(f"Cleaned up namespace: {sid}")
            except Exception as e:
                print(f"Failed to cleanup namespace {sid}: {e}")

threading.Thread(target=cleanup_old_namespaces, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    
