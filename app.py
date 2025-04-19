import streamlit as st
import time
from rag_pipeline import (
    save_pdf,
    process_pdf,
    create_vectorstore,
    retrieve_docs,
    answer_query,
    memory
)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

# --- Page Config ---
st.set_page_config(page_title="⚖️ AI Lawyer", layout="centered")
st.markdown("<h1 style='text-align: center;'>⚖️ AI Lawyer Chatbot</h1>", unsafe_allow_html=True)

# --- Session State Initialization ---
default_keys = {
    "user_queries": [],
    "ai_responses": [],
    "vectorstore": None,
    "chunks": None,
    "pdf_filename": "",
    "summary": ""
}
for key, value in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Upload PDF ---
uploaded_file = st.file_uploader("📄 Upload a legal document (PDF)", type="pdf")

if uploaded_file:
    current_file_name = uploaded_file.name
    if st.session_state.pdf_filename != current_file_name:
        st.success(f"📄 Uploaded: {current_file_name}")
        pdf_path = save_pdf(uploaded_file)
        st.session_state.pdf_filename = current_file_name

        with st.spinner("🔍 Processing document..."):
            chunks = process_pdf(pdf_path)
            vectorstore = create_vectorstore(chunks)

            # ✅ Cache for session
            st.session_state.chunks = chunks
            st.session_state.vectorstore = vectorstore
            st.session_state.summary = ""  # reset summary if new doc
            time.sleep(1)
            st.success("✅ Document ready for querying!")

# --- Summarization ---
if st.button("📜 Summarize Document"):
    if st.session_state.vectorstore:
        with st.spinner("📖 Generating summary..."):
            retrieved = retrieve_docs(st.session_state.vectorstore, "Summarize this document")
            summary = answer_query("Summarize this document", retrieved, memory)
            st.session_state.summary = summary  # ✅ Store summary in session
    else:
        st.error("❌ Please upload a document first.")

# --- Display Summary if available ---
if st.session_state.summary:
    st.markdown("### 📝 Summary:")
    st.markdown(f"<div style='background:#f0f0f0;padding:10px;border-radius:10px'>{st.session_state.summary}</div>", unsafe_allow_html=True)

# --- Chat Interface ---
query = st.text_area("💬 Ask a legal question:", height=100)
if st.button("🔍 Ask AI Lawyer"):
    if not st.session_state.vectorstore:
        st.error("Please upload a document first.")
    elif not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("🤔 Thinking..."):
            docs = retrieve_docs(st.session_state.vectorstore, query)
            response = answer_query(query, docs, memory)

            # Store in chat history
            st.session_state.user_queries.append(query)
            st.session_state.ai_responses.append(response)

# --- Show Full Chat History ---
if st.session_state.user_queries:
    st.markdown("### 🧠 Chat History")
    for q, a in zip(st.session_state.user_queries, st.session_state.ai_responses):
        st.chat_message("user").write(q)
        st.chat_message("AI Lawyer").write(a)

# --- Download Report ---
if st.session_state.user_queries and st.button("📥 Download Q&A Report"):
    pdf_path = "AI_Lawyer_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "AI Lawyer Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Conversation Summary:")

    y = 700
    for q, a in zip(st.session_state.user_queries, st.session_state.ai_responses):
        q_lines = simpleSplit(f"Q: {q}", "Helvetica-Bold", 12, 450)
        a_lines = simpleSplit(f"A: {a}", "Helvetica", 12, 450)
        for line in q_lines + a_lines:
            c.drawString(100, y, line)
            y -= 15
            if y < 50:
                c.showPage()
                y = 750
    c.save()

    with open(pdf_path, "rb") as file:
        st.download_button("📄 Download Report", data=file, file_name="AI_Lawyer_Report.pdf", mime="application/pdf")
