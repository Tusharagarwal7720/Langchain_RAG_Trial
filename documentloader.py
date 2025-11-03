import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, DocxLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from config import DATA_DIR, CHROMA_PERSIST_DIR, EMBEDDING_MODEL

def load_documents(file_paths: list):
    docs = []
    for file in file_paths:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file)
        elif ext == ".txt":
            loader = TextLoader(file)
        elif ext == ".docx":
            loader = DocxLoader(file)
        else:
            print(f"Unsupported file: {file}")
            continue
        docs.extend(loader.load())
    return docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(
        documents, embeddings, persist_directory=CHROMA_PERSIST_DIR
    )
    vectordb.persist()
    return vectordb
