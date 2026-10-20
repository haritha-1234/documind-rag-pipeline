# ============================================================
# MAIN.PY — RUN THE FULL RAG PIPELINE FROM TERMINAL
# ============================================================
# This ties all src/ modules together into one script.
# Run it with: python main.py
# ============================================================

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all our modules
from src.ingestion import load_document
from src.chunking import split_documents
from src.vectorstore import get_vectorstore
from src.retriever import get_retriever, retrieve_chunks
from src.prompt import format_context, build_prompt
from src.generator import load_llm, generate_answer


# ============================================================
# CONFIGURATION
# ============================================================
PDF_PATH = "data/documents/attention_is_all_you_need.pdf"
CHROMA_PATH = ".chroma"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


def run_pipeline(question):
    """
    Run the full RAG pipeline for a question.

    Args:
        question : user's question string

    Returns:
        answer string
    """

    # STEP 1 — load document
    documents = load_document(PDF_PATH)

    # STEP 2 — split into chunks
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # STEP 3 & 4 — embed and store in ChromaDB
    vectorstore = get_vectorstore(chunks, CHROMA_PATH)

    # STEP 5 — create retriever
    retriever = get_retriever(vectorstore, k=TOP_K)

    # STEP 5 — retrieve relevant chunks
    relevant_chunks = retrieve_chunks(retriever, question)

    # STEP 6 — build prompt
    context = format_context(relevant_chunks)
    prompt = build_prompt(context, question)

    # STEP 7 — load LLM and generate answer
    llm = load_llm()
    answer = generate_answer(llm, prompt)

    return answer


if __name__ == "__main__":

    print("="*60)
    print("  DocuMind — RAG Pipeline")
    print("="*60)

    # Load everything ONCE outside the loop
    print("\nInitialising pipeline...")
    documents = load_document(PDF_PATH)
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    vectorstore = get_vectorstore(chunks, CHROMA_PATH)
    retriever = get_retriever(vectorstore, k=TOP_K)
    llm = load_llm()
    print("\nPipeline ready!")

    # Now ask multiple questions without reloading
    questions = [
        "What is the attention mechanism?",
        "What is multi-head attention?",
        "What were the BLEU scores?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")

        # Retrieve, prompt, generate
        relevant_chunks = retrieve_chunks(retriever, question)
        context = format_context(relevant_chunks)
        prompt = build_prompt(context, question)
        answer = generate_answer(llm, prompt)

        print(f"A: {answer}")