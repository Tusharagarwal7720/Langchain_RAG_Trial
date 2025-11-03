from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from config import CHROMA_PERSIST_DIR, LLM_MODEL
from transformers import pipeline

def get_retriever():
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR)
    return vectordb.as_retriever(search_kwargs={"k": 3})

def get_qa_chain():
    retriever = get_retriever()
    llm_pipeline = pipeline("text-generation", model=LLM_MODEL, max_length=200)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
