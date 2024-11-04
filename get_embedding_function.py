from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return embeddings