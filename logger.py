import logging
import os
import warnings

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("legal-rag")

# Suppress noisy logs from external libraries
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)  # suppress model load info
logging.getLogger("chromadb").setLevel(logging.WARNING)               # suppress telemetry
logging.getLogger("pdfplumber").setLevel(logging.ERROR)               # suppress CropBox warnings
logging.getLogger("httpx").setLevel(logging.WARNING)                  # Gemini SDK or requests
# ✅ Suppress specific PDF warnings
warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")


