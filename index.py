# import os 
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from langchain_cohere import CohereEmbeddings
# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# load_dotenv()


# def index_document():

#     # --- step 1 : pdf upload 
    # PDF_PATH = "html_pdf.pdf"
#     pdf_loader = PyPDFLoader(PDF_PATH)
#     raw_docs = pdf_loader.load()
#     print(f"Total documents extracted: {len(raw_docs)}")

#     # --- step 2: chunking 
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 1000,  # 1 chunk has 1000 characters 
#         chunk_overlap = 200 # overlap between two chunks
#     )

#     chunked_docs = text_splitter.split_documents(raw_docs)
#     print(f"Total chunks after splitting: {len(chunked_docs)}")

#     # --- step 3: Embedding model configured 
#     # case 1: Embedding used from gemini AI
#     embeddings = GoogleGenerativeAIEmbeddings(
#         google_api_key = os.environ.get("GEMINI_API_KEY"),
#         model="text-embedding-004"   
#     )
#     # case 2 : Embedding used from Cohere AI
#     # embeddings = CohereEmbeddings(
#     #     model="small",
#     #     cohere_api_key=os.getenv("COHERE_API_KEY")
#     # )

#     # --- step 4 : pinecone handle 
#     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     index_name = os.getenv("PINECONE_INDEX_NAME")
#     pinecone_index = pc.Index(index_name)   # just a handle

#     vectorstore = PineconeVectorStore.from_documents(
#         documents=chunked_docs,
#         embedding=embeddings,
#         index_name=index_name,
#         # maxConcurrency = 5 ,
#     )
#     print("✅ Documents pushed to Pinecone (minimal flow)")

# index_document()


from dotenv import load_dotenv
load_dotenv()  

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
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

    # --- Step 4: Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_index = pc.Index(index_name)   # just a handle

    # --- Step 5: Push chunks to Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=index_name
    )

    print(f"✅ Indexed {len(chunked_docs)} chunks in Pinecone")
