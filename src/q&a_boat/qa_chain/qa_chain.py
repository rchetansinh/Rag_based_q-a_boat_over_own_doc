def answer_question(query, vectorstore, llm, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Answer based on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt)
    print(response.content)
    return response.content
