import os
import pickle
import re
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
from PIL import Image
import pytesseract

PDF_DIR = "compliance_pdfs"
FAISS_INDEX_DIR = "compliance_faiss_index"
METADATA_MAP_FILE = "compliance_doc_metadata.pkl"

def extract_date(text):
    # Matches date formats like 2023-11-23 or 2023/11/23
    match = re.search(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b", text)
    if match:
        return match.group(1)
    return "Unknown"

def ocr_pdf_page(page):
    # Convert pdfplumber page to PIL image and run OCR
    pil_image = page.to_image(resolution=300).original
    text = pytesseract.image_to_string(pil_image)
    return text

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 50:  # Assume valid text
                full_text += text + "\n"
            else:
                # Use OCR fallback for scanned page or images
                ocr_text = ocr_pdf_page(page)
                full_text += ocr_text + "\n"
        return full_text

def determine_factory_id(file_name):
    # Example dynamic factory detection from file name conventions
    if "plantb" in file_name.lower():
        return "PlantB"
    elif "plantc" in file_name.lower():
        return "PlantC"
    else:
        return "PlantA"  # default

def build_vectordb():
    docs = []
    metadata_mapping = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = HuggingFaceEmbeddings()

    for file_name in os.listdir(PDF_DIR):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(PDF_DIR, file_name)
            raw_text = extract_text_from_pdf(file_path)
            extracted_date = extract_date(raw_text)
            factory_id = determine_factory_id(file_name)

            metadata = {
                "source_file": file_name,
                "factory_id": factory_id,
                "document_type": "audit_log",
                "date": extracted_date
            }

            chunks = splitter.split_text(raw_text)
            for chunk in chunks:
                docs.append(Document(page_content=chunk, metadata=metadata))
                metadata_mapping.append(metadata)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_DIR)

    with open(METADATA_MAP_FILE, "wb") as f:
        pickle.dump(metadata_mapping, f)

    print("✅ FAISS vector DB and metadata created!")

if __name__ == "__main__":
    build_vectordb()
