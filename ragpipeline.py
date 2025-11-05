
import os
from langchain_community.llms import GPT4All
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


os.environ["GGML_USE_CUDA"] = "0"
os.environ["GPT4ALL_FORCE_CPU"] = "true"

MODEL_PATH = r"models/phi-2.Q4_K_M.gguf"
CHROMA_DB_DIR = "./chroma_db"


embeddings = None
llm = None
vectorstore = None


def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  
        )
    return embeddings


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=get_embeddings()
        )
    return vectorstore

def close_vectorstore():
   
    global vectorstore
    if vectorstore:
        try:
            vectorstore._client.close()
            print("🧹 Closed Chroma vectorstore connection.")
        except Exception:
            pass
        vectorstore = None


def get_llm():
    global llm
    if llm is None:
        llm = GPT4All(
        model=MODEL_PATH,
        allow_download=False,
        n_predict=2048,  
        temp=0.7
    )
    return llm


from deep_translator import GoogleTranslator
from langdetect import detect

def ask_question(query: str) -> str:

    try:
        detected_lang = detect(query)
    except Exception:
        detected_lang = "en"

   
    if detected_lang == "ja":
        print("Translating query to English...")
        query_en = GoogleTranslator(source="ja", target="en").translate(query)
    else:
        query_en = query

   
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query_en)

    if not docs:
        return "I could not find any relevant information in the uploaded documents."

    
    context = "\n\n".join([d.page_content for d in docs])
    prompt = (
        f"Use the following context to answer the question in simple English.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query_en}\n\nAnswer:"
    )

   
    answer_en = get_llm().invoke(prompt)

    
    if detected_lang == "ja":
        print("Translating answer back to Japanese...")
        answer_final = GoogleTranslator(source="en", target="ja").translate(answer_en)
    else:
        answer_final = answer_en

    return answer_final



def get_qa_chain():
    class SimpleQA:
        def invoke(self, inputs):
            query = inputs.get("query", "")
            return {"result": ask_question(query)}
    return SimpleQA()
