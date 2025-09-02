import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
load_dotenv()


def index_document():

    # --- step 1 : pdf upload 
    PDF_PATH = "kartik.pdf"
    pdf_loader = PyPDFLoader(PDF_PATH)
    raw_docs = pdf_loader.load()
    print(f"Total documents extracted: {len(raw_docs)}")

    # --- step 2: chunking 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,  # 1 chunk has 1000 characters 
        chunk_overlap = 200 # overlap between two chunks
    )

    chunked_docs = text_splitter.split_documents(raw_docs)
    print(f"Total chunks after splitting: {len(chunked_docs)}")

    # --- step 3: Embedding model configured 
    # case 1: Embedding used from gemini AI
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key = os.environ.get("GEMINI_API_KEY"),
        model="text-embedding-004"   
    )
    # case 2 : Embedding used from Cohere AI
    # embeddings = CohereEmbeddings(
    #     model="small",
    #     cohere_api_key=os.getenv("COHERE_API_KEY")
    # )

    # --- step 4 : pinecone handle 
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_index = pc.Index(index_name)   # just a handle

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        index_name=index_name,
        # maxConcurrency = 5 ,
    )
    print("✅ Documents pushed to Pinecone (minimal flow)")

index_document()
