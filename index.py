from dotenv import load_dotenv
load_dotenv()  

import os
import asyncio

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

def get_loader(file_path: str):
    ext = file_path.rsplit(".", 1)[-1].lower()
    loaders = {
        "pdf":  PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt":  TextLoader,
        "csv":  CSVLoader,
        "pptx": UnstructuredPowerPointLoader,
        "xlsx": UnstructuredExcelLoader,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: .{ext}")
    return loaders[ext](file_path)


def index_document(file_path: str, session_id: str):
    print(f"Starting to index document: {file_path}")
    
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        # Step 1: Load PDF from file path
        loader = get_loader(file_path)
        raw_docs = loader.load()
        print(f"Total documents extracted: {len(raw_docs)}")

        # Step 2: Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunked_docs = text_splitter.split_documents(raw_docs)
        print(f"Total chunks after splitting: {len(chunked_docs)}")

        # Step 3: Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-embedding-001",
            output_dimensionality=768
        )


        # Step 4: Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")

        # Step 5: Push chunks to Pinecone
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            index_name=index_name,  
            namespace=session_id
        )

        print(f"✅ Successfully indexed {len(chunked_docs)} chunks in Pinecone")

    except Exception as e:
        print(f"Error in index_document: {e}")
        raise e
