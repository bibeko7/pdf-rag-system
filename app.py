from pathlib import Path

import streamlit as st

from rag.pipeline import RAGPipeline


UPLOAD_DIRECTORY = Path("data/uploads")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background: #0b0d12;
    }

    .main {
        background: #0b0d12;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #171922;
        border-right: 1px solid #292c36;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    /* Main content */
    .block-container {
        max-width: 1200px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Header */
    .brand {
        font-size: 2.8rem;
        font-weight: 750;
        letter-spacing: -1.5px;
        color: #f4f5f7;
        margin-bottom: 0.4rem;
    }

    .subtitle {
        color: #8d929d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Sidebar title */
    .sidebar-title {
        color: #f4f5f7;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }

    .sidebar-section {
        color: #9ca1ad;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 1.5rem;
        margin-bottom: 0.7rem;
    }

    /* Cards */
    .card {
        background: #12151c;
        border: 1px solid #292d38;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
    }

    .answer-card {
        background: #12151c;
        border: 1px solid #292d38;
        border-radius: 14px;
        padding: 1.6rem;
        margin-top: 1.2rem;
    }

    .card-title {
        color: #f1f2f4;
        font-size: 1.05rem;
        font-weight: 650;
        margin-bottom: 0.8rem;
    }

    .muted {
        color: #858b97;
        font-size: 0.9rem;
    }

    /* Status */
    .status {
        background: #111d19;
        border: 1px solid #234b3c;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        color: #8ee0b8;
        margin-bottom: 1.4rem;
    }

    /* Source pills */
    .source-pill {
        display: inline-block;
        background: #1a1e27;
        border: 1px solid #303542;
        border-radius: 8px;
        padding: 0.55rem 0.8rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        color: #c9cdd5;
        font-size: 0.85rem;
    }

    /* Metrics */
    .metric-card {
        background: #12151c;
        border: 1px solid #292d38;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        color: #f4f5f7;
        font-size: 1.45rem;
        font-weight: 700;
    }

    .metric-label {
        color: #858b97;
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
    }

    /* Text input */
    div[data-baseweb="input"] {
        background: #12151c;
        border-radius: 10px;
    }

    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        background: #10131a;
        border: 1px dashed #3a3f4b;
        border-radius: 12px;
    }

    /* Remove excessive top spacing */
    h1, h2, h3 {
        color: #f4f5f7 !important;
    }

    /* -----------------------------------------------------
       RESPONSIVE DESIGN
       Desktop → Tablet → Mobile
       ----------------------------------------------------- */

    /* Prevent horizontal overflow */
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden;
    }

    /* Source pills */
    .source-pill {
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
        margin: 0 0.45rem 0.45rem 0;
    }

    /* Large tablets / small laptops */
    @media (max-width: 1100px) {

        .block-container {
            max-width: 95%;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        .brand {
            font-size: 2.4rem;
        }

        .subtitle {
            font-size: 0.95rem;
        }
    }

    /* Tablets */
    @media (max-width: 900px) {

        .block-container {
            max-width: 100%;
            padding-top: 2rem;
            padding-left: 1.4rem;
            padding-right: 1.4rem;
        }

        .brand {
            font-size: 2.2rem;
            letter-spacing: -1px;
        }

        .subtitle {
            font-size: 0.92rem;
            line-height: 1.5;
        }

        /* Stack metric columns */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.8rem !important;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: calc(50% - 0.4rem) !important;
            flex: 1 1 calc(50% - 0.4rem) !important;
        }

        .card,
        .answer-card {
            padding: 1.2rem;
        }
    }

    /* Mobile */
    @media (max-width: 640px) {

        .block-container {
            width: 100%;
            max-width: 100%;
            padding-top: 1.2rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
            padding-bottom: 2rem;
        }

        .brand {
            font-size: 1.85rem;
            line-height: 1.15;
            letter-spacing: -0.8px;
        }

        .subtitle {
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 1.3rem;
        }

        .card,
        .answer-card {
            border-radius: 11px;
            padding: 1rem;
            margin-bottom: 0.9rem;
        }

        .card-title {
            font-size: 0.98rem;
        }

        .muted {
            font-size: 0.82rem;
        }

        /* One column on mobile */
        [data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 0.7rem !important;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            min-height: 44px;
            font-size: 0.92rem;
        }

        /* Text input */
        div[data-baseweb="input"] {
            width: 100%;
            min-height: 44px;
        }

        div[data-baseweb="input"] input {
            font-size: 16px !important;
        }

        /* File uploader */
        section[data-testid="stFileUploaderDropzone"] {
            width: 100%;
            padding: 0.7rem;
        }

        section[data-testid="stFileUploaderDropzone"] button {
            min-height: 42px;
        }

        /* Source pills */
        .source-pill {
            font-size: 0.78rem;
            padding: 0.48rem 0.65rem;
            margin-right: 0.3rem;
            margin-bottom: 0.35rem;
        }

        /* Status box */
        .status {
            padding: 0.75rem;
            font-size: 0.82rem;
            line-height: 1.5;
        }

        /* Metrics */
        .metric-card {
            width: 100%;
            padding: 0.85rem;
        }

        .metric-value {
            font-size: 1.3rem;
        }

        .metric-label {
            font-size: 0.75rem;
        }

        /* Answer text */
        [data-testid="stMarkdownContainer"] p {
            line-height: 1.6;
        }
    }

    /* Very small phones */
    @media (max-width: 380px) {

        .block-container {
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        .brand {
            font-size: 1.6rem;
        }

        .subtitle {
            font-size: 0.82rem;
        }

        .card,
        .answer-card {
            padding: 0.85rem;
        }

        .source-pill {
            font-size: 0.74rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = None

if "result" not in st.session_state:
    st.session_state.result = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">Document</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Upload a PDF</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:

        st.caption(uploaded_file.name)

        if st.button(
            "Index PDF",
            use_container_width=True,
            type="primary",
        ):

            file_path = (
                UPLOAD_DIRECTORY
                / uploaded_file.name
            )

            with open(file_path, "wb") as file:
                file.write(
                    uploaded_file.getbuffer()
                )

            try:

                with st.spinner(
                    "Indexing document..."
                ):

                    pipeline = RAGPipeline()

                    chunk_count = pipeline.ingest(
                        str(file_path)
                    )

                st.session_state.pipeline = pipeline
                st.session_state.document_name = (
                    uploaded_file.name
                )
                st.session_state.chunk_count = (
                    chunk_count
                )
                st.session_state.result = None

                st.success("Document indexed.")

            except Exception as error:

                st.error(
                    f"Indexing failed: {error}"
                )

    if st.session_state.pipeline is not None:

        st.markdown(
            '<div class="sidebar-section">Current document</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    {st.session_state.document_name}
                </div>
                <div class="muted">
                    {st.session_state.chunk_count} chunks indexed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="brand">PDF RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your documents using Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DOCUMENT STATUS
# ---------------------------------------------------------

if st.session_state.pipeline is not None:

    st.markdown(
        f"""
        <div class="status">
            Document ready: <strong>
            {st.session_state.document_name}
            </strong>
            &nbsp; | &nbsp;
            {st.session_state.chunk_count} chunks indexed
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# QUESTION
# ---------------------------------------------------------

if st.session_state.pipeline is not None:

    st.markdown(
        '<div class="card-title">Ask a question</div>',
        unsafe_allow_html=True,
    )

    question = st.text_input(
        "Question",
        placeholder="What is machine learning?",
        label_visibility="collapsed",
    )

    ask_button = st.button(
        "Ask Question",
        type="primary",
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Searching document and generating answer..."
                ):

                    result = (
                        st.session_state
                        .pipeline
                        .ask(question)
                    )

                st.session_state.result = result

            except Exception as error:

                st.error(
                    f"Question processing failed: {error}"
                )


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

result = st.session_state.result

if result is not None:

    st.markdown(
        '<div class="answer-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card-title">Answer</div>',
        unsafe_allow_html=True,
    )

    st.write(result.answer)

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card-title">Sources</div>',
        unsafe_allow_html=True,
    )

    if result.sources:

        source_html = ""

        for source in result.sources:

            page_number = (
                source.page + 1
                if source.page is not None
                else "Unknown"
            )

            source_html += (
                f'<span class="source-pill">'
                f'Page {page_number}'
                f'</span>'
            )

        st.markdown(
            source_html,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="muted">No source information available.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {len(result.retrieved_documents)}
                </div>
                <div class="metric-label">
                    Retrieved Chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {result.latency_seconds:.2f}s
                </div>
                <div class="metric-label">
                    Response Latency
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.info(
        "Upload and index a PDF from the sidebar to start asking questions."
    )

