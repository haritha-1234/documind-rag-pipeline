# ============================================================
# APP.PY — STREAMLIT UI FOR DOCUMIND RAG PIPELINE
# ============================================================
# This is the visual interface for our RAG pipeline.
# Users can upload a PDF and ask questions about it.
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all our src modules
from src.ingestion import load_document
from src.chunking import split_documents
from src.vectorstore import get_vectorstore
from src.retriever import get_retriever, retrieve_chunks
from src.prompt import format_context, build_prompt
from src.generator import load_llm, generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="DocuMind — RAG Pipeline",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================
st.title("🧠 DocuMind")
st.subheader("Ask questions about your documents using AI")
st.markdown("---")


# ============================================================
# SIDEBAR — CONFIGURATION
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")

    # Chunk size slider
    chunk_size = st.slider(
        "Chunk size",        # label
        min_value=200,       # minimum
        max_value=1000,      # maximum
        value=500,           # default
        step=100,            # step size
        help="How many characters per chunk"
    )

    # Chunk overlap slider
    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="How many characters to overlap between chunks"
    )

    # Top k chunks slider
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help="How many chunks to use as context"
    )

    st.markdown("---")
    st.markdown("**Models used:**")
    st.markdown("🔍 `all-MiniLM-L6-v2`")
    st.markdown("🤖 `flan-t5-small`")
    st.markdown("💾 `ChromaDB`")


# ============================================================
# CACHE EXPENSIVE FUNCTIONS
# ============================================================
# @st.cache_resource means Streamlit runs this only ONCE
# and reuses the result — makes the app much faster!

@st.cache_resource
def load_pipeline(file_path, chunk_size, chunk_overlap, top_k):
    """Load the full RAG pipeline — cached so it runs once."""

    # STEP 1 — load document
    documents = load_document(file_path)

    # STEP 2 — split into chunks
    chunks = split_documents(documents, chunk_size, chunk_overlap)

    # STEP 3 & 4 — embed and store
    chroma_path = ".chroma_app"
    vectorstore = get_vectorstore(chunks, chroma_path)

    # STEP 5 — create retriever
    retriever = get_retriever(vectorstore, k=top_k)

    # STEP 6 & 7 — load LLM
    llm = load_llm()

    return retriever, llm


# ============================================================
# MAIN LAYOUT — TWO COLUMNS
# ============================================================
col1, col2 = st.columns([1, 2])


# ============================================================
# LEFT COLUMN — UPLOAD PDF
# ============================================================
with col1:
    st.header("📄 Upload Document")

    # File uploader — accepts PDF only
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload any PDF document to ask questions about"
    )

    # Use default paper if no file uploaded
    if uploaded_file is None:
        st.info("No file uploaded — using the default Attention paper")
        pdf_path = "data/documents/attention_is_all_you_need.pdf"
        file_ready = os.path.exists(pdf_path)
    else:
        # Save uploaded file to a temp location
        # We need a real file path for PyPDFLoader
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name
        file_ready = True
        st.success(f"Uploaded: {uploaded_file.name}")

    if file_ready:
        st.markdown("---")
        st.markdown("**Pipeline status:**")

        # Show loading spinner while pipeline initialises
        with st.spinner("Loading pipeline..."):
            try:
                retriever, llm = load_pipeline(
                    pdf_path, chunk_size, chunk_overlap, top_k
                )
                st.success("✅ Pipeline ready!")
            except Exception as e:
                st.error(f"Error: {e}")
                retriever, llm = None, None


# ============================================================
# RIGHT COLUMN — CHAT INTERFACE
# ============================================================
with col2:
    st.header("💬 Ask a Question")

    # Initialise chat history in session state
    # session_state persists across reruns of the app
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all previous messages
    for message in st.session_state.messages:

        # Each message has a role (user/assistant) and content
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box at the bottom
    if question := st.chat_input("Ask something about your document..."):

        # Add user question to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Display user question
        with st.chat_message("user"):
            st.markdown(question)

        # Generate and display answer
        with st.chat_message("assistant"):

            if retriever is None or llm is None:
                st.error("Pipeline not ready — please upload a PDF first")
            else:
                # Show spinner while generating
                with st.spinner("Thinking..."):

                    # Run the RAG pipeline
                    relevant_chunks = retrieve_chunks(retriever, question)
                    context = format_context(relevant_chunks)
                    prompt = build_prompt(context, question)
                    answer = generate_answer(llm, prompt)

                    # Show the answer
                    st.markdown(answer)

                    # Show which pages were used as sources
                    pages = [
                        str(c.metadata.get('page', '?'))
                        for c in relevant_chunks
                    ]
                    st.caption(f"📄 Sources: pages {', '.join(pages)}")

                # Add answer to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })