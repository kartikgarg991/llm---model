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
        1: You are Instructor which already analysed the whole PDF and now User ask questions about that pdf . 2: You will given the topK matched data through RAG model searching through vector embeddings and all . 3: Now you will given the user's question and also the relavant context too . 4: If question matches to the context that is provided to you then reply it very clearly and also briefly such that length is justified just only from context not from any outside source . If not the context is there for that question Reply that like the provided question or query is not founded on that document.
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
    You are a query rewriter.
    
    """
    prompt = f"""
    YYou are a query rewriter.
    You will be given the conversation history and current query,
    rewrite the current query such that it is standalone and self-contained and does not require the past conversation .
    DO NOT ANSWER THE QUERY , ONLY RE_WRITE THE QUERY .
    
    History (last 3 turns):
    {history}

    Current query: {query}

    Rewritten query:
    """
    response = chat.send_message(prompt)
    return response.text.strip()