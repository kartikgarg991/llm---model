# import os 
# from dotenv import load_dotenv
# from google import genai
# from google.genai import types
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# load_dotenv()



# # 1. Gemini client
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# chat = client.chats.create(
#     model="gemini-2.5-flash",
#     config= types.GenerateContentConfig(
#         system_instruction= 
#         " 1: You are Instructor which already analysed the whole PDF and now User ask questions about that pdf . 2: You will given the topK matched data through RAG model searching through vector embeddings and all . 3: Now you will given the user's question and also the relavant context too . 4: If question matches to the context that is provided to you then reply it very clearly and also briefly such that length is justified just only from context not from any outside source . If not the context is there for that question Reply that like the provided question or quesry if not founded on that document ?"
#     )
# )

# # 2. Embedding used from gemini AI
# embeddings = GoogleGenerativeAIEmbeddings(
#         google_api_key = os.environ.get("GEMINI_API_KEY"),
#         model="text-embedding-004"   
#     )

# # 3. Pinecone connection 
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = os.getenv("PINECONE_INDEX_NAME")
# index = pc.Index(index_name)  
# vectorstore = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )


# def rewrite_query(history, query):
#     prompt = f"""
#     You are a query rewriter.
#     You will be given the conversation history and current query,
#     rewrite the current query such that it is standalone and self-contained and does not require the past conversation .
#     DO NOT ANSWER THE QUERY , ONLY RE_WRITE THE QUERY .

#     History (last 3 turns):
#     {history}

#     Current query: {query}

#     Rewritten query:
#     """
#     response = chat.send_message(prompt)
#     return response.text.strip()


# # 4. Conversation memory (last 3 turns)
# conversation_history = []

# # 5. Chat-Loop
# while True :
    
#     user_message = input("Enter your query : ").strip()
#     if not user_message:
#         break
    
#     # step1 :- Convert user-query into standard query  
#     prev_conversation = "\n".join(conversation_history[-3:])
#     Standard_query = rewrite_query( prev_conversation , user_message )
#     print(user_message)
#     print(Standard_query)
    
#     # step2 :- Make Standard_query into its vector
#     queryVector = embeddings.embed_query(Standard_query)
    
#     #  step3 :- Query search into Pinecone 
#     raw_results = index.query(
#         vector = queryVector, 
#         top_k = 10, 
#         include_metadata = True
#     )

#     # step4 :- Build Context   
#     matches = raw_results["matches"]
#     if not matches:
#         print("\nAssistant: Sorry, I couldn't find anything in the document for that query.\n")
#         continue
#     context = "\n\n---\n\n".join([m["metadata"]["text"] for m in matches])
    

#     # step5 : Final Prompt
#     final_input = f"""
#     You are an instructor who answers only from the given context.

#     Previous Conversation (last 3 turns):
#     {prev_conversation}
    
#     User Query: {Standard_query}

#     Context (from document):
#     {context}

#     Answer strictly using ONLY the context above. If not relevant, say 'No relevant content found in the document.'
#     """

#     # step6 : Generate Response 
#     response = chat.send_message(final_input)
#     answer = response.text
#     print("\nAssistant:", answer , "\n")

#     # Step 7: Update conversation memory
#     conversation_history.append(f"User: {user_message}\nAssistant: {answer}")
#     conversation_history = conversation_history[-3:]  # Keep last 3 turns

# print("Conversation ended.")



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