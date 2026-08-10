from langchain_ollama import ChatOllama


def get_llm():
    return ChatOllama(model="llama3.1")  # or whatever generation model you've pulled
