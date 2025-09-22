import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max(1, chunk_size - overlap)
    return chunks


def extract_chunks_with_pages(pdf_path, chunk_size=500, overlap=100):
    """Extract page-aware chunks and their page numbers (1-indexed).

    Each PDF page is split independently into overlapping word windows so every
    produced chunk is associated with a single page number.
    """
    doc = fitz.open(pdf_path)
    chunks = []
    pages = []
    for i, page in enumerate(doc):
        words = page.get_text("text").split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            pages.append(i + 1)  # 1-based page index
            start += max(1, chunk_size - overlap)
    return chunks, pages
