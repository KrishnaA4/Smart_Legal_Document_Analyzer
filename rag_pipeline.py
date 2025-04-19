import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from chromadb import Client
from chromadb.config import Settings

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configs
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt Template
custom_prompt_template = """
Use the pieces of information provided in the context and previous conversation history to answer the user's question.
If you don't know the answer, just say that you don't know. Don't make things up.
Only use the information provided in the context.

Previous Conversation:
{history}

Question: {question}
Context:
{context}

Answer:
"""

# Save PDF
def save_pdf(uploaded_file, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return pdf_path

# Load & chunk PDF
def process_pdf(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

# Create in-memory Chroma vector store
def create_vectorstore(chunks):
    client = Client(Settings(anonymized_telemetry=False))
    return Chroma.from_documents(documents=chunks, embedding=embedding_model, client=client)

# Retrieve top-k docs
def retrieve_docs(vectorstore, query, k=5):
    return vectorstore.similarity_search(query, k=k)

# Join context chunks
def build_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Format chat history
def get_chat_history(memory_obj):
    messages = memory_obj.chat_memory.messages if memory_obj else []
    history = ""
    for msg in messages:
        role = "User" if msg.type == "human" else "AI"
        history += f"{role}: {msg.content}\n"
    return history

# Generate answer using prompt + Gemini
def answer_query(query: str, retrieved_docs, memory_obj):
    context = build_context(retrieved_docs)
    history = get_chat_history(memory_obj)
    
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model
    response = chain.invoke({
        "question": query,
        "context": context,
        "history": history
    })

    if memory_obj:
        memory_obj.chat_memory.add_user_message(query)
        memory_obj.chat_memory.add_ai_message(response)

    return response
