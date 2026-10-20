# ============================================================
# CHUNKING MODULE
# ============================================================
# Responsible for splitting documents into smaller chunks.
# Uses RecursiveCharacterTextSplitter which splits on
# paragraphs first, then sentences, then words.
# ============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split a list of Document objects into smaller chunks.

    Args:
        documents   : list of Document objects from ingestion
        chunk_size  : maximum characters per chunk
        chunk_overlap: characters to overlap between chunks

    Returns:
        list of smaller Document chunks
    """

    # Create the text splitter with our settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )

    # Split all documents into chunks
    chunks = text_splitter.split_documents(documents)

    print(f"[Chunking] Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks