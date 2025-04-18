# backend/test/test_vectordb.py

from langchain_huggingface import HuggingFaceEmbeddings
from backend.vector_store.chroma_setup import get_chroma_client

# Initialize client and collection
client = get_chroma_client()
collection = client.get_or_create_collection(name="legal_corpus")

# Initialize the embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Query
query = "What is the legal outcome of cases involving dowry disputes where the parties reached a settlement?"
query_embedding = embedding.embed_query(query)

# Perform similarity search
results = collection.query(query_embeddings=[query_embedding], n_results=3)

# Display results
for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- Result {i+1} ---")
    print(doc)
    print(f"Metadata: {results['metadatas'][0][i]}")
