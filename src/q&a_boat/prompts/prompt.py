from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    prompt = ChatPromptTemplate.from_template(
        """
    Answer the question using only the context below.
    If the answer isn't in the context, say you don't know.
    
    Context:
    {context}
    
    Question:
    {question}
    """
    )
    return prompt
