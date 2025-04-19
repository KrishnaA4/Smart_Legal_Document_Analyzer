import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from vector_database import retrieve_docs as retrieve_filtered_docs
import csv
from datetime import datetime

load_dotenv()

COST_PER_1K_INPUT = 0.0005  # Gemini 1.5 Pro input token cost (USD)
COST_PER_1K_OUTPUT = 0.0015  # Gemini 1.5 Pro output token cost (USD)
CSV_LOG_PATH = "chat_logs.csv"


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
    response = chain.invoke({"question": query, "context": context, "history": history})

    # Extract usage metadata
    usage = getattr(response, "usage_metadata", None)
    input_tokens = usage.get("input_tokens", 0) if usage else 0
    output_tokens = usage.get("output_tokens", 0) if usage else 0
    total_tokens = input_tokens + output_tokens
    cost = round((input_tokens * COST_PER_1K_INPUT + output_tokens * COST_PER_1K_OUTPUT) / 1000, 6)

    # Write to CSV
    log_row = {
        "timestamp": datetime.now().isoformat(),
        "model":"gemini-1.5-pro-002",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost
    }

    file_exists = os.path.isfile(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)

    return response.content if hasattr(response, "content") else str(response)

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
    response = chain.invoke({"context": context})
    
    # Extract usage metadata
    usage = getattr(response, "usage_metadata", None)
    input_tokens = usage.get("input_tokens", 0) if usage else 0
    output_tokens = usage.get("output_tokens", 0) if usage else 0
    total_tokens = input_tokens + output_tokens
    cost = round((input_tokens * COST_PER_1K_INPUT + output_tokens * COST_PER_1K_OUTPUT) / 1000, 6)

    # Write to CSV
    log_row = {
        "timestamp": datetime.now().isoformat(),
        "model":"gemini-1.5-pro-002",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost
    }

    file_exists = os.path.isfile(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)

    return response.content if hasattr(response, "content") else str(response)


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
