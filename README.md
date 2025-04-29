                                                                ⚖️ Smart Legal Document Analysis Platform
An advanced RAG-based AI application that allows uploading legal PDFs, summarizing, asking queries, tracking usage costs, and analyzing API token consumption, built with Google Gemini 1.5 Pro, ChromaDB, and Streamlit.

🚀 Features

Dynamic PDF Upload: Upload any legal document during the session.
Automatic Chunking: PDFs are automatically split into manageable chunks with metadata for better search.
Document Summarization: Summarize the entire uploaded document.
Conversational Chatbot: Ask questions and get contextually accurate legal answers.
Session-specific Memory: Keeps conversation history throughout the session.
RAG Architecture: Retrieval-Augmented Generation — only responds from uploaded content.
Token & Cost Tracker: Logs token usage and API cost details per query.
Analytics Dashboard: View total queries, token consumption, average cost, and more.
Logging: Centralized error and activity logging for debugging and monitoring.

🛠️ Tech Stack

Component	Technology
LLM	Google Gemini 1.5 Pro (via langchain_google_genai)
Embeddings	HuggingFace - MiniLM L6-v2
Vectorstore	ChromaDB (Persistent Client)
Document Loader	PDFPlumberLoader
Chunking	RecursiveCharacterTextSplitter
Backend	Python
Frontend	Streamlit
Logging	Python logging (Rotating File Handler)
Analytics	pandas, matplotlib (optional)


📂 Folder Structure

smart-legal-rag/
├── app.py                  # Streamlit frontend application
├── rag_pipeline.py         # Core RAG logic (summarize, query, analytics)
├── vector_database.py      # Document upload, chunking, vectorstore operations
├── logger.py                # Centralized logging configuration
├── pdfs/                    # Uploaded PDFs
├── vectorstore/db_chroma/   # ChromaDB persistent storage
├── chat_logs.csv            # API token & cost tracking
├── logs/legal_rag.log       # Application logs
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .env                     # Gemini API key (not pushed)


⚡ How to Run
Install Requirements
pip install -r requirements.txt
Set Environment Variables
Create a .env file:
GOOGLE_API_KEY=your-gemini-api-key-here
Start the App
streamlit run app.py


📈 Analytics Dashboard
✅ Track token usage per query.
✅ Track total API cost.
✅ See average tokens and average cost per query.
✅ View full query logs from chat_logs.csv.

🛡️ Security
No external document storage — PDFs and vectors stay local.
Session-based memory — no long-term user data storage.
Token & cost monitoring to avoid unexpected billing.

🎯 Future Improvements
Add user authentication.
Allow uploading multiple PDFs.
Improve prompt engineering with system prompts.
Allow switching between different LLM models dynamically.

👩‍⚖️ Built for
Smart Legal Document Analysis — helping lawyers, researchers, and students instantly summarize and reason over complex legal documents, cost-efficiently and securely.

✨ Thank you for using Smart Legal RAG Platform!
