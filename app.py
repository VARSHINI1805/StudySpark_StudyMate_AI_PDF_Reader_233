import os
import streamlit as st
from dotenv import load_dotenv
import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
import librosa
import soundfile as sf
import numpy as np
from PIL import Image
import io

# Import existing backend modules
from backend.pdf_parser import extract_text_from_pdf, chunk_text
from backend.embedder import Embedder
from backend.retriever import Retriever
from backend.qa_model import answer_question, answer_over_passages

# Load environment variables
load_dotenv()

# -----------------------------
# Model Configuration
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"
DEVICE = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"

# -----------------------------
# Model Loading Functions
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_summarization_model():
    """Load BART summarization model."""
    try:
        return pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            token=HF_TOKEN,
            device=0 if DEVICE == "cuda" else -1
        )
    except Exception as e:
        st.error(f"Failed to load summarization model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_flashcard_model():
    """Load flashcard generation model."""
    try:
        return pipeline(
            "text2text-generation",
            model="grkmkola/flash-cards",
            token=HF_TOKEN,
            device=0 if DEVICE == "cuda" else -1
        )
    except Exception as e:
        st.error(f"Failed to load flashcard model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_study_planning_model():
    """Load T5 study planning model."""
    try:
        return pipeline(
            "text2text-generation",
            model="t5-base",
            token=HF_TOKEN,
            device=0 if DEVICE == "cuda" else -1
        )
    except Exception as e:
        st.error(f"Failed to load study planning model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_grammar_model():
    """Load grammar correction model."""
    try:
        return pipeline(
            "text2text-generation",
            model="deep-learning-analytics/GrammarCorrector",
            token=HF_TOKEN,
            device=0 if DEVICE == "cuda" else -1
        )
    except Exception as e:
        st.error(f"Failed to load grammar model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """Load sentiment analysis model."""
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            token=HF_TOKEN,
            device=0 if DEVICE == "cuda" else -1
        )
    except Exception as e:
        st.error(f"Failed to load sentiment model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load Whisper ASR model."""
    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-base", token=HF_TOKEN)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", token=HF_TOKEN)
        return processor, model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None, None

# -----------------------------
# Stub Functions for Future Models
# -----------------------------
def multi_document_qa(texts, question):
    """Stub function for multi-document Q&A with LayoutLMv3."""
    st.info("🚧 Multi-document Q&A feature coming soon!")
    st.info("This will use microsoft/layoutlmv3-base for advanced document understanding.")
    return "Feature under development"

def ocr_integration(image_file):
    """Stub function for OCR with TrOCR."""
    st.info("🚧 OCR Integration feature coming soon!")
    st.info("This will use microsoft/trocr-large-printed for text extraction from images.")
    return "Feature under development"

def text_to_speech(text):
    """Stub function for TTS with SpeechT5."""
    st.info("🚧 Text-to-Speech feature coming soon!")
    st.info("This will use microsoft/speecht5_tts for audio generation.")
    return "Feature under development"

# -----------------------------
# Processing Functions
# -----------------------------
def process_summarization(text, model):
    """Process text summarization."""
    if not model:
        return "Model not available"
    
    try:
        # Truncate text if too long
        max_length = 1000
        if len(text) > max_length:
            text = text[:max_length]
        
        result = model(text, max_length=150, min_length=30, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"

def process_flashcards(text, model):
    """Process flashcard generation."""
    if not model:
        return "Model not available"
    
    try:
        # Create a prompt for flashcard generation
        prompt = f"Create flashcards for the following text: {text[:500]}"
        result = model(prompt, max_length=200, num_return_sequences=1)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

def process_study_plan(text, model):
    """Process study plan generation."""
    if not model:
        return "Model not available"
    
    try:
        prompt = f"Create a study plan for: {text[:500]}"
        result = model(prompt, max_length=200, num_return_sequences=1)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

def process_grammar_check(text, model):
    """Process grammar correction."""
    if not model:
        return "Model not available"
    
    try:
        result = model(text, max_length=200, num_return_sequences=1)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

def process_sentiment_analysis(text, model):
    """Process sentiment analysis."""
    if not model:
        return "Model not available"
    
    try:
        result = model(text)
        return f"Sentiment: {result[0]['label']} (Confidence: {result[0]['score']:.2f})"
    except Exception as e:
        return f"Error: {str(e)}"

def process_whisper(audio_file, processor, model):
    """Process audio with Whisper."""
    if not processor or not model:
        return "Model not available"
    
    try:
        # Load audio file
        audio_data, sample_rate = librosa.load(audio_file, sr=16000)
        
        # Process with Whisper
        inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
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
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .feature-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        margin: 0.25rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .feature-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .result-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .coming-soon {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="StudyMate AI Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📚"
)

# Custom header
st.markdown("""
<div class="main-header">
    <h1>📚 StudyMate AI Pro</h1>
    <p>Your Advanced Academic Assistant with Hugging Face AI</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h2 style="margin: 0; color: #374151;">⚙ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDFs", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload multiple PDFs to build your knowledge base"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chunking settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 📝 Text Processing")
    chunk_size = st.slider("Chunk size (words)", min_value=200, max_value=1200, value=500, step=50)
    overlap = st.slider("Chunk overlap (words)", min_value=0, max_value=400, value=100, step=10)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Retrieval settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Retrieval Settings")
    top_k = st.slider("Top-K passages", min_value=1, max_value=10, value=3, step=1)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Coming Soon section
    st.markdown('<div class="coming-soon">', unsafe_allow_html=True)
    st.markdown("#### 🚧 Coming Soon")
    st.markdown("""
    - 📄 Multi-document Q&A
    - 🔍 OCR Integration  
    - 🔊 Text-to-Speech
    - 📊 Advanced Analytics
    - 🎯 Smart Recommendations
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Main Content Area
# -----------------------------

# Initialize session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Process uploaded files
if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} PDF(s) uploaded successfully!")
    
    all_chunks = []
    os.makedirs("uploaded_docs", exist_ok=True)
    
    progress = st.progress(0)
    status = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        safe_path = os.path.join("uploaded_docs", uploaded_file.name)
        with open(safe_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        status.write(f"Processing: {uploaded_file.name}")
        text = extract_text_from_pdf(safe_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        progress.progress(idx / len(uploaded_files))
    
    # Generate embeddings and create retriever
    embedder = Embedder()
    embeddings = embedder.get_embeddings(all_chunks)
    st.session_state.chunks = all_chunks
    st.session_state.retriever = Retriever(embeddings, all_chunks)
    
    st.success(f"🎉 Processing complete! {len(all_chunks)} passages ready.")

# Text input area
st.markdown("### 💬 Input Text")
user_input = st.text_area(
    "Enter your text here:",
    placeholder="Paste your study material, questions, or any text you want to analyze...",
    height=150
)

# Feature buttons - Row 1
st.markdown("### 🚀 AI Features")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📘 Summarize", key="summarize"):
        if user_input:
            with st.spinner("Generating summary..."):
                model = load_summarization_model()
                result = process_summarization(user_input, model)
                st.markdown(f"""
                <div class="result-box">
                    <h4>📘 Summary</h4>
                    <p>{result}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to summarize.")

with col2:
    if st.button("🃏 Flashcards", key="flashcards"):
        if user_input:
            with st.spinner("Generating flashcards..."):
                model = load_flashcard_model()
                result = process_flashcards(user_input, model)
                st.markdown(f"""
                <div class="result-box">
                    <h4>🃏 Flashcards</h4>
                    <p>{result}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to generate flashcards.")

with col3:
    if st.button("📅 Study Plan", key="studyplan"):
        if user_input:
            with st.spinner("Creating study plan..."):
                model = load_study_planning_model()
                result = process_study_plan(user_input, model)
                st.markdown(f"""
                <div class="result-box">
                    <h4>📅 Study Plan</h4>
                    <p>{result}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to create a study plan.")

# Feature buttons - Row 2
col4, col5, col6 = st.columns(3)

with col4:
    if st.button("✍️ Grammar Check", key="grammar"):
        if user_input:
            with st.spinner("Checking grammar..."):
                model = load_grammar_model()
                result = process_grammar_check(user_input, model)
                st.markdown(f"""
                <div class="result-box">
                    <h4>✍️ Grammar Check</h4>
                    <p>{result}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to check grammar.")

with col5:
    if st.button("😊 Sentiment Analysis", key="sentiment"):
        if user_input:
            with st.spinner("Analyzing sentiment..."):
                model = load_sentiment_model()
                result = process_sentiment_analysis(user_input, model)
                st.markdown(f"""
                <div class="result-box">
                    <h4>😊 Sentiment Analysis</h4>
                    <p>{result}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze sentiment.")

with col6:
    if st.button("🎤 Voice Input", key="voice"):
        st.info("🎤 Voice input feature - Upload an audio file below")
        audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"])
        if audio_file:
            with st.spinner("Transcribing audio..."):
                processor, model = load_whisper_model()
                result = process_whisper(audio_file, processor, model)
                st.markdown(f"""
                <div class="result-box">
                    <h4>🎤 Transcription</h4>
                    <p>{result}</p>
                </div>
                """, unsafe_allow_html=True)

# Stub features section
st.markdown("### 🚧 Advanced Features (Coming Soon)")
col7, col8, col9 = st.columns(3)

with col7:
    if st.button("📄 Multi-Doc Q&A", key="multidoc"):
        multi_document_qa([], "")

with col8:
    if st.button("🔍 OCR Integration", key="ocr"):
        ocr_integration(None)

with col9:
    if st.button("🔊 Text-to-Speech", key="tts"):
        text_to_speech("")

# Q&A Section (existing functionality)
if st.session_state.retriever:
    st.markdown("### 🤖 Document Q&A")
    question = st.text_input("Ask a question about your uploaded documents:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Searching documents..."):
            embedder = Embedder()
            q_emb = embedder.get_embeddings([question])[0]
            top_chunks = st.session_state.retriever.retrieve(q_emb, top_k=top_k)
            
            passages = [c[0] for c in top_chunks]
            result = answer_over_passages(question, passages)
            answer = result.get("answer", "")
            
            if answer:
                st.markdown(f"""
                <div class="result-box">
                    <h4>🤖 AI Answer</h4>
                    <p>{answer}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No answer found. Try rephrasing your question.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p>StudyMate AI Pro - Powered by Hugging Face Transformers</p>
    <p>Built with ❤️ for students and researchers</p>
</div>
""", unsafe_allow_html=True)
