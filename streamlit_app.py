import os
import streamlit as st
from dotenv import load_dotenv

from backend.pdf_parser import extract_text_from_pdf, chunk_text
from backend.embedder import Embedder
from backend.retriever import Retriever
from backend.qa_model import answer_question, answer_over_passages

# Load environment variables from .env if present
load_dotenv()

# -----------------------------
# Custom CSS for enhanced UI
# -----------------------------
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8faff 0%, #eef2ff 100%);
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 1px solid #34d399;
        color: #065f46;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        color: #92400e;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f8faff 0%, #eef2ff 100%);
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        font-weight: 600;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
    }
    
    /* Sidebar sections */
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Answer section styling */
    .answer-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .answer-section h3 {
        color: #0c4a6e;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Caching utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Cache the SentenceTransformer-backed embedder instance."""
    return Embedder(model_name)

@st.cache_data(show_spinner=False)
def compute_embeddings(texts: list[str], model_name: str):
    """Cache embeddings for given texts + model to avoid recomputation."""
    emb = get_embedder(model_name).get_embeddings(texts)
    return emb

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="StudyMate AI", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📚"
)

# Custom header
st.markdown("""
<div class="main-header">
    <h1>📚 StudyMate AI</h1>
    <p>Your Intelligent Academic Assistant</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar controls (Enhanced UI)
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h2 style="margin: 0; color: #374151;">⚙ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)

    # Embedding & chunking controls in styled container
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🔧 Model Settings")
    
    embedder_model = st.selectbox(
        "📊 Embedding Model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
        index=0,
        help="🔍 Retriever: sentence-transformers/all-MiniLM-L6-v2",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 📝 Text Processing")
    chunk_size = st.slider("📏 Chunk size (words)", min_value=200, max_value=1200, value=500, step=50)
    overlap = st.slider("🔗 Chunk overlap (words)", min_value=0, max_value=400, value=100, step=10)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Retrieval Settings")
    top_k = st.slider("🔢 Top-K passages", min_value=1, max_value=10, value=3, step=1)
    show_scores = st.checkbox("📊 Show similarity scores", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🤖 AI Model Info")
    st.info("🧠 Reader: deepset/roberta-base-squad2\n✅ No external API key required")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    if st.button("🧹 Clear Session", type="secondary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# File uploader with enhanced styling
# -----------------------------
st.markdown("### 📁 Document Upload")
st.markdown('<div class="info-card">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "📄 Upload one or more PDFs", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="💡 Tip: Upload multiple PDFs to create a comprehensive knowledge base"
)
st.markdown('</div>', unsafe_allow_html=True)

# Keep some state across interactions
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Process uploads with enhanced feedback
# -----------------------------
if uploaded_files:
    st.markdown(f"""
    <div class="info-card success-card">
        <h4 style="margin: 0;">✅ Upload Successful</h4>
        <p style="margin: 0.5rem 0 0 0;">{len(uploaded_files)} PDF(s) uploaded and ready for processing</p>
    </div>
    """, unsafe_allow_html=True)

    all_chunks: list[str] = []
    os.makedirs("uploaded_docs", exist_ok=True)

    progress = st.progress(0)
    status = st.empty()

    # Track page numbers per chunk
    all_pages: list[int] = []

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        # Save locally to ensure PyMuPDF can open it reliably
        safe_path = os.path.join("uploaded_docs", uploaded_file.name)
        with open(safe_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        status.write(f"🔍 Extracting text from: *{uploaded_file.name}*")
        # Page-aware chunking
        status.write(f"✂ Chunking: *{uploaded_file.name}* (size={chunk_size}, overlap={overlap})")
        try:
            from backend.pdf_parser import extract_chunks_with_pages
            chunks, pages = extract_chunks_with_pages(safe_path, chunk_size=chunk_size, overlap=overlap)
        except Exception:
            # Fallback to old behavior if page-aware function not available
            text = extract_text_from_pdf(safe_path)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            pages = [None] * len(chunks)
        all_chunks.extend(chunks)
        all_pages.extend(pages)

        progress.progress(idx / len(uploaded_files))

    status.write("🚀 Generating embeddings…")
    embeddings = compute_embeddings(all_chunks, embedder_model)

    # Build retriever and store in session
    st.session_state.chunks = all_chunks
    st.session_state.pages = all_pages
    st.session_state.retriever = Retriever(embeddings, all_chunks, all_pages)

    st.markdown(f"""
    <div class="info-card">
        <h4 style="margin: 0; color: #059669;">🎉 Processing Complete!</h4>
        <p style="margin: 0.5rem 0 0 0;">Prepared <strong>{len(all_chunks)}</strong> passages for intelligent retrieval</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Q&A Section with enhanced design
# -----------------------------
st.markdown("### 💬 Ask Your Question")

question_container = st.container()
with question_container:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    question = st.text_input(
        "🤔 Enter your question here:", 
        placeholder="e.g., What are the main concepts discussed in the documents?",
        help="💡 Ask specific questions for better results"
    )
    st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 🎯 Answer Generation")
    if st.button("💡 Get Answer", type="primary") and question:
        if not st.session_state.retriever:
            st.markdown("""
            <div class="info-card warning-card">
                <h4 style="margin: 0;">⚠ Knowledge Base Missing</h4>
                <p style="margin: 0.5rem 0 0 0;">Please upload PDFs first to build the knowledge base.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Compute query embedding and retrieve top passages
            q_emb = get_embedder(embedder_model).get_embeddings([question])[0]
            top_chunks = st.session_state.retriever.retrieve(q_emb, top_k=top_k)

            # Run reader per passage to avoid truncation and improve accuracy
            passages = [c[0] for c in top_chunks]
            with st.spinner("🧠 Running AI reader (deepset/roberta-base-squad2)…"):
                result = answer_over_passages(question, passages)
            answer, score, best_idx = result.get("answer", ""), result.get("score", 0), result.get("passage_index", -1)

            # Derive page number for the best passage if available
            page_label = None
            if 0 <= best_idx < len(top_chunks):
                _, page_num, _ = top_chunks[best_idx]
                if page_num is not None:
                    page_label = f" | 📄 Page {page_num}"
                else:
                    page_label = ""
            else:
                page_label = ""

            # Show answer with enhanced styling and page number
            if answer:
                st.markdown(f"""
                <div class="answer-section">
                    <h3>✅ AI Answer</h3>
                    <p style="font-size: 1.1rem; line-height: 1.6;">{answer}</p>
                    <small style="color: #0c4a6e;">🎯 Confidence: {score:.2f} | 📄 Source: Passage {best_idx + 1}{page_label}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-card warning-card">
                    <h4 style="margin: 0;">🤷‍♂ No Answer Found</h4>
                    <p style="margin: 0.5rem 0 0 0;">The model couldn't find a confident answer. Try adjusting Top-K settings or rephrasing your question.</p>
                </div>
                """, unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "passages": top_chunks,
            })

            # Download options with enhanced styling
            if answer:
                st.download_button(
                    label="⬇ Download Answer",
                    data=answer,
                    file_name="studymate_answer.txt",
                    mime="text/plain",
                )

with col2:
    st.markdown("#### 🔍 Retrieved Context")
    if "retriever" in st.session_state and st.session_state.retriever and question:
        q_emb = get_embedder(embedder_model).get_embeddings([question])[0]
        top_chunks_preview = st.session_state.retriever.retrieve(q_emb, top_k=top_k)
        
        for i, (chunk, page_num, dist) in enumerate(top_chunks_preview, start=1):
            page_text = f" | 📄 Page {page_num}" if page_num is not None else ""
            score_text = f" | 📊 Score: {dist:.3f}" if show_scores else ""
            with st.expander(f"📄 Passage {i}{page_text}{score_text}"):
                st.write(chunk)
        
        # Enhanced download button for context
        context_text = "\n\n---\n\n".join([c for c, _, _ in top_chunks_preview])
        st.download_button(
            label="⬇ Download Context",
            data=context_text,
            file_name="studymate_context.txt",
            mime="text/plain",
        )

# -----------------------------
# Enhanced History Section
# -----------------------------
with st.expander("🕘 Conversation History", expanded=False):
    if st.session_state.history:
        st.markdown("### 📚 Previous Q&A Sessions")
        for i, item in enumerate(reversed(st.session_state.history), start=1):
            st.markdown(f"""
            <div class="info-card">
                <h4 style="margin: 0; color: #374151;">❓ Question {i}</h4>
                <p style="margin: 0.5rem 0;"><em>{item['question']}</em></p>
                <h4 style="margin: 1rem 0 0 0; color: #059669;">✅ Answer {i}</h4>
                <p style="margin: 0.5rem 0 0 0;">{item['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if item.get("passages"):
                with st.expander(f"📄 Referenced passages for Q{i}"):
                    for j, tup in enumerate(item["passages"], start=1):
                        # Support both legacy (chunk, score) and new (chunk, page, score)
                        if isinstance(tup, (list, tuple)) and len(tup) == 3:
                            chunk, page_num, dist = tup
                        elif isinstance(tup, (list, tuple)) and len(tup) == 2:
                            chunk, dist = tup
                            page_num = None
                        else:
                            # Fallback: just show the text
                            chunk, page_num, dist = str(tup), None, None

                        page_text = f" (Page {page_num})" if page_num is not None else ""
                        st.markdown(f"📄 Passage {j}{page_text}:** {chunk}")
                        if show_scores and dist is not None:
                            st.caption(f"📊 Similarity score: {dist:.3f}")
    else:
        st.markdown("""
        <div class="info-card">
            <h4 style="margin: 0; color: #6b7280;">📝 No History Yet</h4>
            <p style="margin: 0.5rem 0 0 0;">Ask a question above to start building your conversation history!</p>
        </div>
        """, unsafe_allow_html=True)