from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import traceback
import tempfile
from supabase import create_client

from index import index_document
from query import chat, embeddings, pinecone_index, rewrite_query, conversation_history



SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


app = Flask(__name__)

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
    print("upload rquest received")
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({"error": "No file"}), 400

        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("No file selected")
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        file_bytes = file.read()

         # --- Upload to Supabase Storage (bucket = pdfs)
        result = supabase.storage.from_("pdfs").upload(filename, file_bytes)
        print("File uploaded to Supabase successfully")

        # Download back from Supabase
        pdf_bytes = supabase.storage.from_("pdfs").download(filename)
        print(f"Downloaded {len(pdf_bytes)} bytes from Supabase")


       # Create temporary file in Render's temp directory
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                temp_file_path = tmp.name
                print(f"Created temp file: {temp_file_path}")

            # Process the file
            index_document(temp_file_path)
            print("Indexing complete")
            
            return jsonify({"success": True, "filename": filename})

        finally:
            # Always clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                print(f"Cleaned up temp file: {temp_file_path}")

    except Exception as e:
        print("Upload error:", e)  # <--- This will show the error in your terminal
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
    prev_conv = "\n".join(conversation_history[-5:])
    std_query = rewrite_query(prev_conv, query)

    print( f"Original Query: {query}")
    print( f"Rewritten Query: {std_query}")



    # Get vector & query Pinecone
    query_vector = embeddings.embed_query(std_query)
    raw_results = pinecone_index.query(vector=query_vector, top_k=10, include_metadata=True)
    matches = raw_results["matches"]

    if not matches:
        answer = "Sorry, I couldn't find anything in the document for that query."
    else:
        context = "\n\n---\n\n".join([m["metadata"]["text"] for m in matches])
        prompt = f"""
        "You are an instructor who answers only from the given context. Write the answer in plain text, no headings, no markdown, no formatting."
        
        Previous Conversation (last 5 turns):
        {prev_conv}

        User Query: 
        {std_query}

        Context (from document):
        {context}

        Answer strictly using ONLY the context above. If not relevant, say 'No relevant content found in the document.'
        Also make answer in mutiple paragraphs as required so that user can understand. Give answer in proper spacing . Dont respond answer in cluster where the space and paragraph needed .
        """
        answer = chat.send_message(prompt).text

    # Update conversation history
    conversation_history.append(f"User: {query}\nAssistant: {answer}")
    conversation_history[:] = conversation_history[-5:]

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

