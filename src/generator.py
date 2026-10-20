# ============================================================
# GENERATOR MODULE
# ============================================================
# Responsible for loading the LLM and generating answers.
# Uses a local HuggingFace model — no API key needed.
# ============================================================

from transformers import pipeline


# The local model we use for generation
LLM_MODEL = "google/flan-t5-small"


def load_llm():
    """
    Load the local LLM pipeline.
    Downloads the model on first run, then uses cache.

    Returns:
        HuggingFace pipeline object
    """

    print(f"[Generator] Loading local LLM: {LLM_MODEL}")

    # pipeline() loads model and tokenizer together
    llm = pipeline(
        "text2text-generation",  # task type for flan-t5
        model=LLM_MODEL,
        max_new_tokens=200
    )

    print(f"[Generator] LLM ready!")
    return llm


def generate_answer(llm, prompt):
    """
    Generate an answer from the LLM given a prompt.

    Args:
        llm    : loaded HuggingFace pipeline
        prompt : complete prompt string from prompt.py

    Returns:
        answer string
    """

    # Run the model on the prompt
    result = llm(prompt)

    # Extract the generated text from the result
    answer = result[0]["generated_text"]

    return answer