# ============================================================
# INGESTION MODULE
# ============================================================
# Responsible for loading documents from disk into memory.
# Currently supports PDF files using LangChain's PyPDFLoader.
# Can be extended to support txt, docx, web pages etc.
# ============================================================

from langchain_community.document_loaders import PyPDFLoader


def load_document(file_path):
    """
    Load a PDF document from disk.
    Returns a list of Document objects — one per page.
    """

    # Create the loader with the file path
    loader = PyPDFLoader(file_path)

    # Load all pages into memory
    documents = loader.load()

    print(f"[Ingestion] Loaded {len(documents)} pages from {file_path}")
    return documents