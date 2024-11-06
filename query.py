import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.retrievers import BM25Retriever

from groq import Groq
import os
from get_embedding_function import get_embedding_function
import dotenv
from typing import List, Tuple
from chromadb import Client
from chromadb.config import Settings
dotenv.load_dotenv("env.txt", override=True)

CHROMA_PATH = os.environ.get("CHROMA_PATH")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""



def query_rag(query_text: str, history: List[Tuple[str, str]]):
    
    # Load the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    existing_item = db.get(include=["documents", "metadatas"])

    documents = []
    for text, metadata in zip(existing_item['documents'], existing_item['metadatas']):
        documents.append(Document(page_content=text, metadata=metadata))
    print(documents)
    bm25_retriever = BM25Retriever.from_documents(documents)
    #results = db.similarity_search_with_score(query_text, k=5)
    results = bm25_retriever.get_relevant_documents(query_text, n=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = os.environ.get("MODEL")
    client = Groq(api_key=os.environ.get("GROK_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.01,
        top_p=0.1,
    )
    content = response.choices[0].message.content
    #print('content: ',content)
    sources = [doc.page_content for doc in results]
    formatted_response = f"Response: {content} \nSources: {sources}"
    #print(formatted_response)
    history.append((query_text, formatted_response))

    return history