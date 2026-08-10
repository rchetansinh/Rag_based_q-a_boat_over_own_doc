def retrieve(vector_db):
    return vector_db.as_retriever()
    # return retriever.invoke("what is list ?")
