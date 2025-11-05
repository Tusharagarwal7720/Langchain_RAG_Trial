import os
import shutil
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_chroma import Chroma  


def load_documents(file_paths: list):
    docs = []
    for file in file_paths:
        ext = os.path.splitext(file)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file)
        elif ext == ".txt":
            loader = TextLoader(file)
        elif ext == ".docx":
            loader = Docx2txtLoader(file)
        else:
            print(f"Unsupported file type: {file}")
            continue
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} document sections from {len(file_paths)} file(s).")
    return docs



def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks



def create_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
   
    print(" Vectorstore created successfully at 'chroma_db'.")
    return vectordb



def delete_vectorstore():
    if os.path.exists("chroma_db"):
        for attempt in range(3):
            try:
                shutil.rmtree("chroma_db")
                print("Existing vectorstore deleted successfully.")
                return
            except PermissionError:
                print(f"Access denied while deleting 'chroma_db' (attempt {attempt+1}/3). Retrying...")
                time.sleep(1)
            except Exception as e:
                print(f"Unexpected error while deleting vectorstore: {e}")
                break
        print("Could not delete 'chroma_db' — it might still be locked by another process.")
    else:
        print("No existing vectorstore found.")
