from chromadb import PersistentClient

def get_chroma_client(persist_path="chroma_db"):
    return PersistentClient(path=persist_path)
