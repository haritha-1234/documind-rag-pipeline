# ============================================================
# PROMPT MODULE
# ============================================================
# Responsible for building the prompt that gets sent to
# the LLM. Combines retrieved context with the question.
# ============================================================


def format_context(chunks):
    """
    Join retrieved chunks into one context string.

    Args:
        chunks : list of Document objects from retriever

    Returns:
        single string with all chunk text joined together
    """

    # Join all chunk texts with double newline between them
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    return context


def build_prompt(context, question):
    """
    Build the full prompt to send to the LLM.
    Combines system instruction, context and question.

    Args:
        context  : formatted string of retrieved chunks
        question : user's question

    Returns:
        complete prompt string ready for the LLM
    """

    prompt = f"""Answer the question based only on the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""

    return prompt