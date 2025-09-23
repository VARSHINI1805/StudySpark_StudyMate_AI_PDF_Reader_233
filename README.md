# StudyMate AI 🧠

<p align="center">
  <strong>Your intelligent academic assistant for effortless document Q&A.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  <a href="https://streamlit.io" target="_blank"><img src="https://img.shields.io/badge/Built%20with-Streamlit-ff69b4.svg" alt="Built with Streamlit"></a>
  <img src="https://img.shields.io/badge/Model-RoBERTa-667eea.svg" alt="RoBERTa Model">
</p>

---

Tired of endlessly scrolling through hundreds of pages in your PDFs to find a single piece of information? **StudyMate AI** transforms your static documents into a dynamic, searchable knowledge base. Upload your study materials, research papers, or textbooks, and ask questions in plain English to get instant, accurate answers sourced directly from the text.

<!-- 💡 TIP: Record a short GIF of your app in action and replace the placeholder below! -->
<p align="center">
  <img src="https://raw.githubusercontent.com/your-username/your-repo/main/assets/app_demo.gif" alt="CogniRead App Demo" width="800"/>
</p>

## ✨ Key Features

-   **🧠 Intelligent Q&A**: Ask complex questions and get precise answers, not just keyword matches.
-   **📚 Multi-PDF Knowledge Base**: Combine multiple documents to create a unified source of truth.
-   **🤖 Local & Private**: Runs entirely on your machine using powerful open-source models. No data leaves your computer, and no API keys are required.
-   **🎯 Source-Referenced Answers**: Every answer is backed by the exact passage from your documents, complete with page numbers.
-   **⚙️ Advanced Customization**: Fine-tune the text processing and retrieval settings directly from the UI for optimal results.
-   **📜 Conversation History**: Keep track of your Q&A sessions for easy reference.
-   **🎨 Modern UI**: A clean, intuitive, and responsive interface built with a custom Streamlit theme.

## ⚙️ How It Works

CogniRead AI uses a **Retrieval-Augmented Generation (RAG)** pipeline to provide answers:

1.  **📄 Ingestion & Chunking**: PDFs are parsed, and their text is broken down into smaller, overlapping chunks.
2.  **🧠 Embedding**: Each chunk is converted into a numerical vector (an embedding) using a Sentence Transformer model (`all-MiniLM-L6-v2`). This captures its semantic meaning.
3.  **🔍 Retrieval**: When you ask a question, it's also embedded. The system then performs a vector search to find the text chunks most semantically similar to your question.
4.  **📖 Answer Generation**: The retrieved chunks (the context) and your question are passed to a QA model (`roberta-base-squad2`), which reads the context to find and extract the final answer.

## 🛠️ Tech Stack

| Component         | Technology                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| **Application**   | Streamlit                                                                     |
| **PDF Parsing**   | PyMuPDF                                                            |
| **Embedding**     | Sentence-Transformers (`all-MiniLM-L6-v2`)                                     |
| **QA Model**      | Hugging Face Transformers (`deepset/roberta-base-squad2`)        |
| **Core ML**       | PyTorch, NumPy                                            |

## 🚀 Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/VARSHINI1805/CogniRead-AI.git
cd CogniRead-AI
```

### 2. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to keep dependencies isolated.
```sh
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required packages from `requirements.txt`.
```sh
pip install -r requirements.txt
```

### 4. Run the Application
Launch the Streamlit app with the following command:
```sh
streamlit run streamlit_app.py
```
The application will open automatically in your web browser. Now you're ready to upload your documents and start asking questions!

