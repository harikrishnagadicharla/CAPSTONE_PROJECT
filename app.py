import os
import pickle
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from fpdf import FPDF

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyDJnXjUqMrR4txh3z1U29Gzpqb0nGo2vJg"))

FAISS_INDEX_DIR = "compliance_faiss_index"
METADATA_FILE = "compliance_doc_metadata.pkl"

model = genai.GenerativeModel("gemini-1.5-flash")

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    return db, metadata

db, metadata_list = load_db()

st.title("📑 Industrial Compliance Assistant (RAG + Gemini)")
st.markdown("Ask questions based on compliance audit reports.")

query = st.text_input("🔍 Ask your question:")

def clean_markdown_headers(text):
    text = re.sub(r'#+ ', '', text)
    text = re.sub(r'\*\*(.*?)\*\*:', r'\1:', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    return text

def format_result_for_display(text):
    text = re.sub(r'#+ ', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = text.replace("\n", "<br>")
    return text

def save_response_to_pdf(response_text, filename="gemini_compliance_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ✅ Add DejaVu font
    font_path = r"C:\Users\91798\Downloads\DejaVuSans.ttf"  # Make sure this path is correct
    if not os.path.exists(font_path):
        raise FileNotFoundError("DejaVuSans.ttf not found. Download and place it in your working directory.")

    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 12)

    for line in response_text.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(filename)
    return filename

def get_compliance_response(query, k=5):
    relevant_keywords = [
        "audit", "compliance", "non-conformance", "iso", "recommendation",
        "corrective", "plant", "report", "deadline", "NC", "standard",
        "summary", "factory", "flag"
    ]
    if not any(word in query.lower() for word in relevant_keywords):
        return """
❌ **Out-of-Scope Question Detected**

This assistant is designed to help analyze and summarize **compliance audit reports** (e.g., ISO 9001, non-conformities, recommendations, deadlines).

Your question appears to be unrelated to that scope.

🔎 Try asking:
- What were the non-compliance flags in the last audit?
- What corrective actions were suggested?
- What is the compliance percentage for Plant A?
"""

    docs = db.similarity_search(query, k=k)
    context_texts = []
    metadata_texts = []

    for d in docs:
        context_texts.append(d.page_content)
        meta = d.metadata
        metadata_texts.append(
            f"- File: {meta.get('source_file')}, Type: {meta.get('document_type')}, "
            f"Date: {meta.get('date')}, Factory: {meta.get('factory_id')}"
        )

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

if query:
    with st.spinner("🔍 Analyzing documents and generating structured response..."):
        result = get_compliance_response(query)

        st.markdown("### 📘 Gemini's Compliance Report")
        cleaned_result = format_result_for_display(result)
        st.markdown(cleaned_result, unsafe_allow_html=True)

        try:
            pdf_path = save_response_to_pdf(result)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Report as PDF", f, file_name="compliance_report.pdf")
        except Exception as e:
            st.error(f"PDF Generation Error: {e}")

with st.expander("💡 Example Questions You Can Ask"):
    st.markdown("""
- What corrective actions were recommended, and what are their deadlines?
- List the non-compliance flags for ISO 9001.
- What are the deadlines for corrective actions?
- Are there any unresolved compliance flags from previous audits?
- What recommendations were made in the last audit?
- How many compliance issues were found?
""")
