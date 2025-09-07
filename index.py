from dotenv import load_dotenv
load_dotenv()  

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pinecone
from langchain_pinecone import PineconeVectorStore

import asyncio
def index_document(PDF_PATH):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # --- Step 1: Load PDF
    pdf_loader = PyPDFLoader(PDF_PATH)
    raw_docs = pdf_loader.load()
    print(f"Total documents extracted: {len(raw_docs)}")

    # --- Step 2: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunked_docs = text_splitter.split_documents(raw_docs)
    print(f"Total chunks after splitting: {len(chunked_docs)}")

    # --- Step 3: Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv("GEMINI_API_KEY"),
        model="text-embedding-004"
    )

    # # --- Step 4: Initialize Pinecone
    # pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # index_name = os.getenv("PINECONE_INDEX_NAME")
    # pinecone_index = pc.Index(index_name)   # just a handle

    # # --- Step 5: Push chunks to Pinecone
    # vectorstore = PineconeVectorStore.from_documents(
    #     documents=chunked_docs,
    #     embedding=embeddings,
    #     index_name=index_name
    # )
    # --- Step 4: Initialize Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT") 
    )
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_index = pinecone.Index(index_name)  
    
    # --- Step 5: Push chunks to Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=index_name
    )

    print(f"✅ Indexed {len(chunked_docs)} chunks in Pinecone")
