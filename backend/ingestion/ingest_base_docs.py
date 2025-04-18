# backend/ingestion/ingest_base_docs.py

import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from backend.vector_store.chroma_setup import get_chroma_client
from backend.utils.pdf_loader import load_pdf_text

# Initialize embeddings and text splitter
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Load ChromaDB
client = get_chroma_client()
collection = client.get_or_create_collection(name="legal_corpus")

# Load PDF files from folder
doc_folder = Path("legal_docs/persistent")
pdf_files = list(doc_folder.glob("*.pdf"))

for pdf_file in pdf_files:
    filename = pdf_file.name

    # Check if already exists
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        print(f"⚠️ Skipping {filename} — already ingested.")
        continue

    print(f"📄 Processing: {filename}")
    text = load_pdf_text(str(pdf_file))
    chunks = splitter.split_text(text)

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename}] * len(chunks)
    embeddings = embedding.embed_documents(chunks)

    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"✅ Ingested: {filename}")

print("✅ Done! Base legal docs added to 'legal_corpus'.")
