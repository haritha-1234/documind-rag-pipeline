# ============================================================
# RETRIEVER MODULE
# ============================================================
# Responsible for searching the vector store and returning
# the most relevant chunks for a given question.
# ============================================================


def get_retriever(vectorstore, k=3):
    """
    Create a retriever from a vector store.

    Args:
        vectorstore : Chroma vector store object
        k           : number of chunks to return per question

    Returns:
        LangChain retriever object
    """

    retriever = vectorstore.as_retriever(
        # similarity = find most similar chunks to the question
        search_type="similarity",
        search_kwargs={"k": k}
    )

    print(f"[Retriever] Retriever ready — returning top {k} chunks")
    return retriever


def retrieve_chunks(retriever, question):
    """
    Retrieve the most relevant chunks for a question.

    Args:
        retriever : LangChain retriever object
        question  : user's question as a string

    Returns:
        list of relevant Document chunks
    """

    # Search the vector store for relevant chunks
    chunks = retriever.invoke(question)

    print(f"[Retriever] Found {len(chunks)} relevant chunks")
    return chunks