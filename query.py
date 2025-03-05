import torch
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_milvus import Milvus
from typing import List, Tuple
from utils import load_config

# Check if CUDA is available and set device accordingly
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# ----------------------
# Vector Store and Retriever
# ----------------------
config = load_config("config.yaml")
MILVUS_PATH = config["vectordb"]["path"]
EMBED_MODEL_ID = config["embedding"]["embed_model_id"]
collection_name = config["vectordb"]["collection_name"]
milvus_uri = os.path.join(MILVUS_PATH, "docling.db")  # or set as needed

# Specify device for embeddings
embedding = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_ID,
    model_kwargs={"device": device}
)
vector_store = Milvus(
    embedding,
    collection_name=collection_name,
    connection_args={"uri": milvus_uri},
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

#----------------------
#LLM Model Setup
#----------------------
LOCAL_PATH = config['llm']['llm_path']  # Path where the model is stored

# Configure model to use the same device
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with explicit device map
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_PATH, 
    quantization_config=bnb_config, 
    device_map="auto"  # Let transformers handle device placement
)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH,device_map="auto")
# Set pad token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# Create pipeline with device specified
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=512,
    eos_token_id=terminators,
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
# ----------------------
# RAG Prompt Setup
# ----------------------
prompt_rag_template = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions using provided context.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_rag_template,
)

qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}, return_source_documents=False
)

# ----------------------
# Functions for Actions
# ----------------------

def query_rag(query_text: str, history: List[Tuple[str, str]]):
    """Queries the RAG system and updates history."""
    result = qa_chain.invoke({"query": query_text})  # ✅ Ensure the key is correct
    answer = result.get("result", "No result found.")
    
    # Store in history
    history.append({"role": "user", "content": query_text})
    history.append({"role": "assistant", "content": answer})

    return history



