# Test_Script.py
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# --- Use FLAN-T5 (smart, concise) ---
MODEL_NAME = "google/flan-t5-small"

print(f"Loading {MODEL_NAME}...")

pipe = pipeline(
    "text2text-generation",
    model=MODEL_NAME,
    max_new_tokens=50,
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Ask ---
question = "What is the capital of France?"
print(f"\nQ: {question}")
print(f"A: {llm.invoke(question)}")