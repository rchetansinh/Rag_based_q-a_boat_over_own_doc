from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def splitting_text(documents) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(documents)
