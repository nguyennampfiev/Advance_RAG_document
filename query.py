import torch
import os
from typing import List, Tuple
from utils import load_config
from model import ModelFactory


# Check if CUDA is available and set device accordingly
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# ----------------------
# Vector Store and Retriever
# ----------------------
config = load_config("config.yaml")

#----------------------
#LLM Model Setup
#---------------------
# Create model
model = ModelFactory.create_model(config,context_window=5)


# ----------------------
# Functions for Actions
# ----------------------

def query_rag(query_text: str, history: List[Tuple[str, str]]):
    """Queries the RAG system and updates history."""
    history = model.query_rag(query_text, history)
    
    return history, ""



