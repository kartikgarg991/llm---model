import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
load_dotenv()

# --- 1. Gemini client ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="""
        You are an instructor who answers questions strictly using provided context.
        Context will come from user's uploaded PDFs via Pinecone search.
        """
    )
)

# --- 2. Embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=os.getenv("GEMINI_API_KEY"),
    model="text-embedding-004"
)

# --- 3. Pinecone index handle ---

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_index = pc.Index(index_name) 
vectorstore = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name=index_name
    )

# --- 4. Conversation memory ---
conversation_history = []

# --- 5. Query rewriter function ---
def rewrite_query(history, query):
    """
    Rewrites a user query into a standalone, self-contained query.
    """
    prompt = f"""
    You are a query rewriter.
    History (last 3 turns):
    {history}

    Current query: {query}

    Rewritten query:
    """
    response = chat.send_message(prompt)
    return response.text.strip()