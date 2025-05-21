# Import necessary libraries
import os, re, pickle
import pdfplumber, pytesseract
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define directories and file names
PDF_DIR = "compliance_pdfs"
FAISS_INDEX_DIR = "compliance_faiss_index"
METADATA_FILE = "compliance_doc_metadata.pkl"

# Extracts the first date (YYYY-MM-DD or YYYY/MM/DD) from text
def extract_date(text):
    match = re.search(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b", text)
    return match.group(1) if match else "Unknown"

# Extracts all text from a PDF file using direct extraction or OCR fallback
def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(
            p.extract_text() if p.extract_text() and len(p.extract_text().strip()) > 50 
            else pytesseract.image_to_string(p.to_image(resolution=300).original)
            for p in pdf.pages
        )

# Dynamically determine the factory ID based on file name
def get_factory_id(name):
    name = name.lower()
    return "PlantB" if "plantb" in name else "PlantC" if "plantc" in name else "PlantA"

# --- Build Vector DB ---
def build_vectordb():
    docs, metadata_list = [], []

    # Initialize chunk splitter and embedding model
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = HuggingFaceEmbeddings()

    # Iterate through each PDF file in the directory
    for file in os.listdir(PDF_DIR):
        if not file.endswith(".pdf"):
            continue

        # Extract raw text and generate metadata
        text = extract_text_from_pdf(os.path.join(PDF_DIR, file))
        metadata = {
            "source_file": file,
            "factory_id": get_factory_id(file),
            "document_type": "audit_log",
            "date": extract_date(text)
        }

        # Split text into chunks and create documents with metadata
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata=metadata))
            metadata_list.append(metadata)

    # Build FAISS vector store and save locally
    FAISS.from_documents(docs, embeddings).save_local(FAISS_INDEX_DIR)

    # Save metadata mapping for later use
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata_list, f)

    print("✅ Vector DB and metadata saved.")

# Run the function
if __name__ == "__main__":
    build_vectordb()
