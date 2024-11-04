import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from groq import Groq
import os
from get_embedding_function import get_embedding_function
import dotenv
from typing import List, Tuple

dotenv.load_dotenv("env.txt", override=True)

CHROMA_PATH = os.environ.get("CHROMA_PATH")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

LLAMA3_8B_INSTRUCT = "llama-3.1-8b-instant"


def query_rag(query_text: str, history: List[Tuple[str, str]]):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=2)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
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
    sources = [doc.page_content for doc, _score in results]
    formatted_response = f"Response: {content} \nSources: {sources}"
    #print(formatted_response)
    history.append((query_text, content))

    return history