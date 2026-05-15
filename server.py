from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import traceback
import tempfile
import threading 
from index import index_document
from query import chat, rewrite_query, search_context
from session_store import clear_session, get_expired_sessions, get_history, mark_cleanup_if_due, save_history, touch_session

app = Flask(__name__)

CLEANUP_COOLDOWN_SECONDS = int(os.getenv("CLEANUP_COOLDOWN_SECONDS", "3600"))
_cleanup_lock = threading.Lock()


def cleanup_expired_namespaces():
    from pinecone import Pinecone

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    expired = get_expired_sessions()
    for sid in expired:
        try:
            index.delete(delete_all=True, namespace=sid)
            clear_session(sid)
            print(f"Cleaned up namespace: {sid}")
        except Exception as e:
            print(f"Failed to cleanup namespace {sid}: {e}")


def run_cleanup_job():
    if not _cleanup_lock.acquire(blocking=False):
        return

    try:
        cleanup_expired_namespaces()
    except Exception as e:
        print(f"Cleanup failed: {e}")
    finally:
        _cleanup_lock.release()


def trigger_cleanup_from_homepage():
    try:
        if not mark_cleanup_if_due(CLEANUP_COOLDOWN_SECONDS):
            return
    except Exception as e:
        print(f"Could not schedule cleanup: {e}")
        return

    threading.Thread(target=run_cleanup_job, daemon=True).start()


@app.route("/")
def home():
    trigger_cleanup_from_homepage()
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    session_id = request.form.get("session_id")
    if not session_id:
        return jsonify({"error": "No session_id"}), 400

    touch_session(session_id)


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
            
        touch_session(session_id)
        query = data.get("query")
        if not query:
            return jsonify({"error": "No query"}), 400

        history = get_history(session_id)
        prev_conv = "\n".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in history
        ])
        print(f"[DEBUG] Session ID: {session_id}")
        print(f"[DEBUG] History for this session: {history}")
        print(f"[DEBUG] Previous conversation passed to rewriter:\n{prev_conv}")
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
            Format the answer clearly using Markdown:
            - If multiple facts are present, use bullet points.
            - Put each important fact on a separate line.
            - Use bold Markdown section headings when helpful.
            - If exact URLs are present in the context, format them as Markdown links.
            - If only link labels are present and exact URLs are missing, mention the labels as plain text.
            - Do not invent or guess URLs.
            - Keep short answers concise.
            - Do not write one long paragraph when listing details.
            """
            answer = chat.send_message(prompt).text

        history.append({"user": query, "assistant": answer})
        save_history(session_id, history)

        return jsonify({"answer": answer})

    except Exception as e:
        traceback.print_exc()
        error_text = str(e)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
            return jsonify({"error": "Gemini API limit exhausted. Please try again after some time."}), 429
        return jsonify({"error": error_text}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    
