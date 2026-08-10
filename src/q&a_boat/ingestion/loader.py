from langchain_community.document_loaders import PyPDFLoader
import os
from logger import get_logger

# Initialize logger for this specific file
logger = get_logger(__name__)


def load_pdf_documents():
    pdf_path = os.getenv("PDF_FILE")
    logger.info(f"Pdf file path : {pdf_path}")
    pages = []
    if not pdf_path:
        logger.error("Pdf file enviroment veriable not found.")
        return []

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # logger.info(f"Successfully loaded {len(pages)} pages.")
    except Exception as e:
        logger.error(f"Failed to parse Pdf file:{e}")
        return []
    return pages
