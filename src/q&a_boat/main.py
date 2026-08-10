from ingestion import loader
from logger import get_logger
from chunking import chunker
from embeddings import embeddings
from vectordb import cromadb
from retrivel import retrivel
from llm import llms
from prompts import prompt
from qa_chain import qa_chain
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=r"C:\Users\Pragneshsinh.Rathod\Downloads\All_Projects\Git_Repo_Python\Rag_based_Q&A_boat_over_own_doc\Rag_based_Q&A_boat_over_own_doc\config\.env"
)

logger = get_logger(__name__)

st.title("RAG Q&A over your documents Pragnesh")


@st.cache_resource
def build_pipeline():
    logger.info("Starting RAG workflow processing...")

    pdf_path = os.getenv("PDF_FILE")  # same env var your loader uses
    model = embeddings.ollama_embeddings()

    # Only load + chunk if we actually need to re-embed
    from vectordb.cromadb import _get_file_hash
    import os as _os, json as _json

    metadata_file = "./chroma_db/source_metadata.json"
    needs_embedding = True
    if _os.path.exists(metadata_file):
        with open(metadata_file) as f:
            saved = _json.load(f)
        needs_embedding = saved.get("hash") != _get_file_hash(pdf_path)

    if needs_embedding:
        document_pages = loader.load_pdf_documents()
        final_split_documents = chunker.splitting_text(document_pages)
    else:
        final_split_documents = None  # not needed, will just load existing store

    vectordb = cromadb.store_chromadb(final_split_documents, model, pdf_path)
    llm = llms.get_llm()
    return vectordb, llm


with st.spinner("Indexing documents... (first run only)"):
    vectordb, llm = build_pipeline()

st.success("Ready! Ask a question below.")

query = st.chat_input("Ask about your documents")
if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.answer_question(query, vectordb, llm)
    st.write(answer)


# from  ingestion import loader
# from logger import get_logger
# from chunking import chunker
# from embeddings import embeddings
# from vectordb import cromadb
# from retrivel import retrivel
# from llm import llms
# from prompts import prompt
# from qa_chain import qa_chain
# import streamlit as st
#
#
# logger = get_logger(__name__)
# def main():
#     print("Starting RAG workflow")
#     logger.info("Starting RAG workflow processing...")
#     document_pages =loader.load_pdf_documents()
#     logger.info(f"Main execution received {len(document_pages)} pages for indexing.")
#     #logger.info(document_pages)
#     logger.info("starting splitting documents...")
#     final_split_documents = chunker.splitting_text(document_pages)
#     #logger.info(final_split_documents)
#     logger.info(f"Splitting documents completed.")
#     logger.info("Starting embeding documents...")
#     emded_documents,model = embeddings.ollama_embeddings(final_split_documents)
#     #logger.info(emded_documents)
#     logger.info(f"Embedding documents completed.")
#     logger.info("Starting vectorizing documents...")
#     vectordb = cromadb.store_chromadb(final_split_documents,model)
#     doc = vectordb.similarity_search("basic PDF functionality?")
#     logger.info(doc[0].page_content)
#     logger.info(f"Vectorizing documents completed.")
#     #s = vectordb.similarity_search_with_score("basic PDF functionality?")
#     logger.info("Start retrieving embeddings...")
#     retriver_info = retrivel.retrieve(vectordb)
#     #logger.info(retriver_info)
#     logger.info(f"Retrieving embeddings completed.")
#     logger.info("Retrieving LLM")
#     llm = llms.get_llm()
#     logger.info("Got LLM")
#     logger.info("Getting promts Now")
#     prompt_ = prompt.get_prompt()
#     logger.info("Got prompt")
#     logger.info("Start gettting answer")
#     query = "how is pdf ? "
#     answer = qa_chain.answer_question(query,vectordb,llm)
#     logger.info(f"Got answer :{answer}")
#
#     @st.cache_resource
#     def init():
#         vs = vectordb
#         model = llm
#         return vs, model
#
#     vectorstore, llm = init()
#
#     query = st.chat_input("Ask about your documents")
#     if query:
#         answer = qa_chain.answer_question(query, vectorstore, llm)
#         st.write(answer)
#
# if __name__ == "__main__":
#     main()
#
