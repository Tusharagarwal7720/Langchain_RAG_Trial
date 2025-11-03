
from langchain_ollama import Ollama 
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

llm = Ollama(model="llama2", n_ctx=512)  


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def get_qa_chain():
    return RetrievalQA(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )
