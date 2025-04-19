import streamlit as st
import time
from vector_database import index_pdf, upload_pdf
from rag_pipeline import answer_query, retrieve_docs, llm_model, summarize_document, generate_report

st.set_page_config(page_title="⚖️ AI Lawyer", layout="centered")
st.markdown("""
    <h1 style='text-align: center;'>⚖️ AI Lawyer Chatbot</h1>
    <style>
        body {
            background-color: #f8f9fa;
            color: #333;
            font-family: Arial, sans-serif;
        }
        .stTextArea textarea {
            font-size: 16px;
            padding: 10px;
            background-color: #fff;
            color: #000;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        .stButton button {
             background-color: #546e7a; /* Elegant gray */
             color: white;
              border-radius: 8px;
         padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease;
        }
.stButton button:hover {
    background-color: #455a64; /* Darker shade on hover */
}

        .summary-box {
            background-color: #f0f0f0;
            padding: 10px;
            border-left: 4px solid #0d6efd;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if "user_queries" not in st.session_state:
    st.session_state.user_queries = []
if "ai_responses" not in st.session_state:
    st.session_state.ai_responses = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "pdf_filename" not in st.session_state:
    st.session_state.pdf_filename = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a legal document (PDF)", type="pdf")

if uploaded_file:
    current_file_name = uploaded_file.name
    if st.session_state.pdf_filename != current_file_name:
        st.success(f"📄 Uploaded: {current_file_name}")
        file_path = upload_pdf(uploaded_file)
        st.session_state.pdf_filename = current_file_name

        with st.spinner("🔍 Processing document..."):
            index_pdf(file_path)  # indexes into Chroma by file-based collection name
            time.sleep(1)
            st.success("✅ Document ready for querying!")

        # Clear session history when switching document
        st.session_state.user_queries = []
        st.session_state.ai_responses = []
        st.session_state.summary = ""

# Summarize
if st.button("📜 Summarize Document"):
    if uploaded_file:
        with st.spinner("📖 Generating summary..."):
            docs = retrieve_docs("Summarize this document", uploaded_file.name)
            summary = summarize_document(docs)
            st.session_state.summary = summary
    else:
        st.error("❌ Please upload a document first.")

if st.session_state.summary:
    st.markdown("### 📝 Summary:")
    st.markdown(f"<div class='summary-box'>{st.session_state.summary}</div>", unsafe_allow_html=True)

# Chat Interface
query = st.text_area("💬 Ask a legal question:", height=100)
if st.button("🔍 Ask AI Lawyer"):
    if not uploaded_file:
        st.error("❌ Please upload a document first.")
    elif not query.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("🤔 Thinking..."):
            docs = retrieve_docs(query, uploaded_file.name)
            response = answer_query(docs, llm_model, query)
            st.chat_message("user").write(query)
            st.chat_message("AI Lawyer").write(response)
            st.session_state.user_queries.append(query)
            st.session_state.ai_responses.append(response)

# Chat History
if st.session_state.user_queries:
    st.markdown("### 🧠 Chat History")
    for q, a in zip(st.session_state.user_queries, st.session_state.ai_responses):
        st.chat_message("user").write(q)
        st.chat_message("AI Lawyer").write(a)

# Report download
if st.session_state.user_queries and st.button("📥 Download Q&A Report"):
    report_path = generate_report(st.session_state.user_queries, st.session_state.ai_responses)
    with open(report_path, "rb") as file:
        st.download_button("📄 Download Report", data=file, file_name="AI_Lawyer_Report.pdf", mime="application/pdf")
