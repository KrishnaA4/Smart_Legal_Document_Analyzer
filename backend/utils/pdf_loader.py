from langchain_community.document_loaders import PyPDFLoader

def load_pdf_as_documents(pdf_path: str):
    loader = PyPDFLoader(file_path=pdf_path)
    return loader.load()
 # returns List[Document] with rich metadata
