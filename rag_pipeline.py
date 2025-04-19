import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from vector_database import retrieve_docs as retrieve_filtered_docs

load_dotenv()

llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

def retrieve_docs(query, file_name):
    return retrieve_filtered_docs(query, file_name)

def get_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)

custom_prompt_template = """
Use the pieces of information provided in the context and previous conversation history to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context.

Previous Conversation:
{history}

Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query, history=""):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context, "history": history})

def summarize_document(documents):
    context = get_context(documents)
    summary_prompt = """
    Summarize the given legal document concisely while preserving key details.
    Provide a structured summary that highlights the most important points.

    Document:
    {context}

    Summary:
    """
    prompt = ChatPromptTemplate.from_template(summary_prompt)
    chain = prompt | llm_model
    return chain.invoke({"context": context})

def generate_report(user_queries, ai_responses):
    pdf_path = "AI_Lawyer_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "AI Lawyer Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Below is a record of your conversation with AI Lawyer.")

    y = 700
    for q, a in zip(user_queries, ai_responses):
        q_lines = simpleSplit(f"Q: {q}", "Helvetica-Bold", 12, 450)
        a_lines = simpleSplit(f"A: {a}", "Helvetica", 12, 450)
        for line in q_lines + a_lines:
            c.drawString(100, y, line)
            y -= 15
            if y < 50:
                c.showPage()
                y = 750
    c.save()
    return pdf_path
