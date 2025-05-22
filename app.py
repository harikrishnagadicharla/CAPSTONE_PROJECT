import os
import pickle
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import google.generativeai as genai
from fpdf import FPDF
from datetime import datetime
import tempfile

# --- Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyDJnXjUqMrR4txh3z1U29Gzpqb0nGo2vJg"))  # Use .env for safety
FAISS_INDEX_DIR = "compliance_faiss_index"
METADATA_FILE = "compliance_doc_metadata.pkl"

model = genai.GenerativeModel("gemini-1.5-flash")
embeddings = HuggingFaceEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# --- UI Setup ---
st.title("📑 Industrial Compliance Assistant")
st.markdown("Upload files or metadata and ask compliance-related questions.")

# --- Load or Initialize DB ---
@st.cache_resource
def load_or_create_db():
    if os.path.exists(FAISS_INDEX_DIR):
        db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "rb") as f:
                metadata = pickle.load(f)
        else:
            metadata = []
    else:
        db = FAISS.from_documents([], embeddings)
        metadata = []
    return db, metadata

db, metadata_list = load_or_create_db()

# --- File Upload Section ---
st.header("📤 Upload Files")
uploaded_files = st.file_uploader("Upload PDFs or Excel Sheets", accept_multiple_files=True, key="file_upload")
if uploaded_files:
    new_documents = []
    for file in uploaded_files:
        filename = file.name
        filetype = filename.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        if filetype == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif filetype in ["xls", "xlsx"]:
            loader = UnstructuredExcelLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {filename}")
            continue

        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)

        for doc in split_docs:
            doc.metadata.update({
                "source_file": filename,
                "document_type": filetype.upper(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "factory_id": "ManualUpload",
                "custom_note": ""
            })

        new_documents.extend(split_docs)

    if new_documents:
        db.add_documents(new_documents)
        metadata_list.extend([doc.metadata for doc in new_documents])

        db.save_local(FAISS_INDEX_DIR)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(metadata_list, f)

        st.success("✅ Files uploaded and indexed successfully!")

# --- Manual Metadata Input Section ---
st.header("🌐 Enter Metadata Directly")
manual_meta = st.text_area("Enter audit notes, summaries, or links (treated as searchable text)", key="meta_input")
if manual_meta:
    doc = Document(page_content=manual_meta, metadata={
        "source_file": "ManualMetadataEntry",
        "document_type": "TEXT",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "factory_id": "MetaEntry",
        "custom_note": "UserInput"
    })
    db.add_documents([doc])
    metadata_list.append(doc.metadata)
    db.save_local(FAISS_INDEX_DIR)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata_list, f)
    st.success("✅ Metadata saved successfully!")

# --- Agents ---
def retriever_agent(query, k=5):
    return db.similarity_search(query, k=k)

def metadata_extractor_agent(docs):
    context_texts, metadata_texts = [], []
    for d in docs:
        context_texts.append(d.page_content)
        meta = d.metadata
        metadata_texts.append(
            f"- File: {meta.get('source_file')}, Type: {meta.get('document_type')}, "
            f"Date: {meta.get('date')}, Factory: {meta.get('factory_id')}, Note: {meta.get('custom_note', '')}"
        )
    return context_texts, metadata_texts

def compliance_analysis_agent(query, context_texts, metadata_texts):
    prompt = f"""
You are a regulatory compliance assistant.

Using the following document context and metadata, extract:
1. Extracted Entities – Key structured data such as compliance standard, date, factory.
2. Compliance Flags – Highlight non-compliance issues or sections.
3. Audit Summary – Short summary with % compliance, recommendations, and deadlines.

Metadata:
{chr(10).join(metadata_texts)}

Document Context:
{chr(10).join(context_texts)}

Question: {query}

Respond in the following format:
---
**Extracted Entities:** <structured key info>

**Compliance Flags:** <list of non-compliance flags>

**Audit Summary:** <summary with compliance % and actions>
---
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"

def format_result_for_display(text):
    text = re.sub(r'#+ ', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = text.replace("\n", "<br>")
    return text

def save_response_to_pdf(response_text, filename="gemini_compliance_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError("DejaVuSans.ttf not found.")

    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 12)

    for line in response_text.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(filename)
    return filename

def handle_user_query(query):
    keywords = ["audit", "compliance", "non-conformance", "iso", "recommendation", "corrective", "plant", "report", "deadline", "NC", "standard", "summary", "factory", "flag"]
    if not any(word in query.lower() for word in keywords):
        return """
❌ **Out-of-Scope Question Detected**

This assistant is designed to help analyze and summarize **compliance audit reports**.

🔎 Try asking:
- What were the non-compliance flags in the last audit?
- What corrective actions were suggested?
- What is the compliance percentage for Plant A?
"""
    docs = retriever_agent(query)
    context_texts, metadata_texts = metadata_extractor_agent(docs)
    return compliance_analysis_agent(query, context_texts, metadata_texts)

# --- Query Section ---
st.header("🔍 Ask Your Compliance Question")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Analyzing your documents..."):
        result = handle_user_query(query)
        st.markdown("### 📘 Gemini's Compliance Report")
        cleaned_result = format_result_for_display(result)
        st.markdown(cleaned_result, unsafe_allow_html=True)

        try:
            pdf_path = save_response_to_pdf(result)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Report as PDF", f, file_name="compliance_report.pdf")
        except Exception as e:
            st.error(f"PDF Generation Error: {e}")

# --- Examples ---
with st.expander("💡 Example Questions You Can Ask"):
    st.markdown("""
- What corrective actions were recommended, and what are their deadlines?
- List the non-compliance flags for ISO 9001.
- What are the deadlines for corrective actions?
- Are there any unresolved compliance flags from previous audits?
- What recommendations were made in the last audit?
- How many compliance issues were found?
""")
