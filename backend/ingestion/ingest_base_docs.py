
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from backend.vector_store.chroma_setup import get_chroma_client
from backend.utils.pdf_loader import load_pdf_as_documents
from dotenv import load_dotenv

load_dotenv()

# Embedding function using HuggingFace model (MiniLM is fast and good)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

client = get_chroma_client()
collection = client.get_or_create_collection(name="legal_corpus")

doc_folder = Path("legal_docs/persistent")
pdf_files = list(doc_folder.glob("*.pdf"))

for pdf_path in pdf_files:
    filename = pdf_path.name

    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        print(f"⚠️ Skipping {filename} — already ingested.")
        continue

    print(f"📄 Processing: {filename}")
    documents = load_pdf_as_documents(str(pdf_path))  # returns LangChain Documents with metadata
    split_docs = splitter.split_documents(documents)

    ids = [f"{filename}_{i}" for i in range(len(split_docs))]
    texts = [doc.page_content for doc in split_docs]
    metadatas = [{"source": filename}] * len(split_docs)


    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    print(f"✅ Ingested: {filename}")

print("📦 Persisting ChromaDB...")

print("✅ All documents ingested successfully.")
