from langchain_community.document_loaders import PyPDFLoader,UnstructuredPDFLoader
import os
from langchain_core.documents import Document
from logger import get_logger
from chunking import chunker
from qa_chain import qa_chain


# Initialize logger for this specific file
logger = get_logger(__name__)

import tiktoken
from langchain_core.documents import Document

# Initialize tokenizer for precise token counting (e.g., cl100k_base for tiktoken/Ollama models)
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_token_count(text: str) -> int:
    return len(tokenizer.encode(text, disallowed_special=()))

def load_pdf_documents(pdf_path: str = None):
    if pdf_path is None:
        pdf_path = os.getenv("PDF_FILE")
    loader = UnstructuredPDFLoader(
        pdf_path,
        mode="elements",
        strategy="hi_res", #hi_res strategy detects tables/images as distinct elements and can extract table structure. 
        #Requires unstructured[pdf] plus poppler and tesseract installed locally.
        extract_images_in_pdf= True,       # saves images to disk
        infer_table_structure=True,       # keeps table structure in metadata as HTML
    )
    raw_docs = loader.load()
    text_docs, table_docs, image_docs = [], [], []

    # Categorize incoming unstructured elements
    for doc in raw_docs:
        category = doc.metadata.get("category")
        if category == "Table":
            html_table = doc.metadata.get("text_as_html", doc.page_content)
            
            # Fallback if HTML generation failed
            if not html_table or not html_table.strip():
                html_table = doc.page_content
                
            doc.page_content = html_table
            table_docs.append(doc)
        elif category == "Image":
            image_docs.append(doc)
        else:
            text_docs.append(doc)

    # 1. Process Standard Text C1hunks
    text_chunks = chunker.splitting_text(text_docs)

    # 2. Process Tables (Precise token checking to prevent Ollama crashes)
    final_table_docs = []
    for t in table_docs:
        # nomic-embed-text max context is 8192, but keeping chunks around 512-1024 tokens is optimal
        if get_token_count(t.page_content) > 1000:
            logger.warning(f"Table element is too large ({get_token_count(t.page_content)} tokens). Splitting safely.")
            
            # Chunk the table but explicitly propagate and adjust chunk metadata
            split_chunks = chunker.splitting_text([t])
            for i, chunk in enumerate(split_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["is_table_fragment"] = True
            final_table_docs.extend(split_chunks)
        else:
            t.metadata["is_table_fragment"] = False
            final_table_docs.append(t)

    # 3. Process Images (With clean text truncation boundaries)
    image_path_key = "image_path"
    processed_image_docs = []

    if image_docs:
      logger.info(f"Found {len(image_docs)} image(s). Metadata keys: {list(image_docs[0].metadata.keys())}")

    for img in image_docs:
        path = img.metadata.get(image_path_key) or img.metadata.get("filename")
        if not path:
            logger.warning(f"No image path found. Keys available: {list(img.metadata.keys())}")
            continue
        try:
            description = qa_chain.vision_model(path)

            if get_token_count(description) > 1000:
                logger.info("Image description too long. Truncating text smoothly.")
                cutoff_desc = description[:4000]
                last_period = cutoff_desc.rfind('.')
                if last_period > 0:
                    description = cutoff_desc[:last_period + 1] + " [Truncated]"
                else:
                    description = cutoff_desc + "... [Truncated]"

            processed_image_docs.append(
                Document(
                    page_content=f"[IMAGE DESCRIPTION]\nSource File: {path}\n{description}",
                    metadata={**img.metadata, "source_image": path},
                )
            )
        except Exception as e:
            logger.warning(f"Skipping image {path}: {e}")

    # Return safe, bound-checked data to Chroma
    return text_chunks + final_table_docs + processed_image_docs


# def load_pdf_documents():
#     pdf_path = os.getenv("PDF_FILE")
#     loader = UnstructuredPDFLoader(
#     pdf_path,
#     mode="elements",
#     strategy="hi_res", #hi_res strategy detects tables/images as distinct elements and can extract table structure. 
#     #Requires unstructured[pdf] plus poppler and tesseract installed locally.
#     extract_images_in_pdf=True,       # saves images to disk
#     infer_table_structure=True,       # keeps table structure in metadata as HTML
#     )
#     raw_docs = loader.load()
#     text_docs, table_docs, image_docs = [], [], []

#     for doc in raw_docs:
#         category = doc.metadata.get("category")
#         logger.info(f"Catogory is : {category}")
#         if category == "Table":
#             # HTML representation of table structure lives here
#             html_table = doc.metadata.get("text_as_html", doc.page_content)
#             #table = doc.metadata.get("text", doc.page_content)
#             doc.page_content = html_table  # keep structure instead of flattened text
#             table_docs.append(doc)
#         elif category == "Image":
#             image_docs.append(doc)
#         else:
#             text_docs.append(doc)

#     # 1. Text: chunk normally with splitter
#     text_chunks = chunker.splitting_text(text_docs)

#     # 2. Tables: keep as whole documents (don't chunk — splitting breaks table meaning)
#     #  Optionally prefix with a short description so retrieval matches better
#     for t in table_docs:
#         t.page_content

#     # 3. Images: run each through a local vision model, then treat the
#     #    description as a text chunk tied back to the image path
#     image_path_key = "./chroma_db/image_path"  # unstructured stores this in metadata
#     processed_image_docs = []
#     for img in image_docs:
#         path = img.metadata.get(image_path_key)
#         if path:
#             description = qa_chain.vision_model(path)  # your own function
#             processed_image_docs.append(
#                 Document(
#                     page_content=f"[IMAGE DESCRIPTION]\n{description}",
#                     metadata={**img.metadata, "source_image": path},
#                 )
#             )

#     # Merge everything and hand off to your existing embedding + vector store step
#     return text_chunks + table_docs + processed_image_docs


#this will work when we have only text in pdf 
# def load_pdf_documents():
#     pdf_path = os.getenv("PDF_FILE")
#     logger.info(f"Pdf file path : {pdf_path}")
#     pages = []
#     if not pdf_path:
#         logger.error("Pdf file enviroment veriable not found.")
#         return []

#     try:
#         loader = PyPDFLoader(pdf_path)
#         pages = loader.load()
#         # logger.info(f"Successfully loaded {len(pages)} pages.")
#     except Exception as e:
#         logger.error(f"Failed to parse Pdf file:{e}")
#         return []
#     return pages
