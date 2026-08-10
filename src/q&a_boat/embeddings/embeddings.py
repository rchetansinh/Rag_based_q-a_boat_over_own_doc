from langchain_ollama import OllamaEmbeddings


def ollama_embeddings():
    # Just return the model — don't pre-embed documents here,
    # Chroma.from_documents will call the model itself when needed
    return OllamaEmbeddings(model="nomic-embed-text")
