# ============================================================
# VECTOR STORE MODULE
# ============================================================
# Responsible for creating and loading the ChromaDB
# vector database. Handles both creating a fresh database
# and loading an existing one from disk.
# ============================================================

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# The embedding model we use throughout the project
# Must be the same model for both storing and searching!
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the HuggingFace embedding model.
    Runs locally — no API key needed.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"[VectorStore] Embedding model loaded: {EMBEDDING_MODEL_NAME}")
    return embedding_model


def create_vectorstore(chunks, chroma_path):
    """
    Create a fresh ChromaDB vector store from chunks.
    Embeds all chunks and saves to disk.

    Args:
        chunks      : list of Document chunks from chunking
        chroma_path : where to save the database on disk

    Returns:
        Chroma vector store object
    """

    # Load the embedding model
    embedding_model = load_embedding_model()

    print(f"[VectorStore] Creating vector store at {chroma_path}...")

    # Embed all chunks and save to ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=chroma_path
    )

    print(f"[VectorStore] Stored {vectorstore._collection.count()} chunks")
    return vectorstore


def load_vectorstore(chroma_path):
    """
    Load an existing ChromaDB vector store from disk.
    Much faster than recreating — no re-embedding needed.

    Args:
        chroma_path : path where the database was saved

    Returns:
        Chroma vector store object
    """

    # Load the embedding model
    embedding_model = load_embedding_model()

    print(f"[VectorStore] Loading existing vector store from {chroma_path}...")

    # Connect to existing database
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_model
    )

    print(f"[VectorStore] Loaded {vectorstore._collection.count()} chunks")
    return vectorstore


def get_vectorstore(chunks, chroma_path):
    """
    Smart loader — loads existing store if available,
    otherwise creates a fresh one.

    Args:
        chunks      : chunks to use if creating fresh
        chroma_path : path to the database

    Returns:
        Chroma vector store object
    """

    if os.path.exists(chroma_path):
        # Database already exists — just load it
        return load_vectorstore(chroma_path)
    else:
        # No database yet — create fresh
        return create_vectorstore(chunks, chroma_path)