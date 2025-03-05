import os
import torch
from typing import Dict, Any
from transformers import (
    pipeline, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_milvus import Milvus

from .base_model import AbstractRAGModel

class LlamaRAGModel(AbstractRAGModel):
    """
    Concrete implementation of RAG model using Hugging Face models.
    """
    def __init__(self, config: Dict[str, Any], context_window: int = 5):
        super().__init__(config, context_window)
    def _setup_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Set up Hugging Face embeddings based on configuration.
        
        :return: Configured embedding model
        """
        return HuggingFaceEmbeddings(
            model_name=self.config["embedding"]["embed_model_id"],
            model_kwargs={"device": self.device}
        )
    
    def _setup_vector_store(self) -> Milvus:
        """
        Set up Milvus vector store based on configuration.
        
        :return: Configured vector store
        """
        milvus_uri = os.path.join(
            self.config["vectordb"]["path"], 
            "docling.db"
        )
        
        return Milvus(
            self.embedding,
            collection_name=self.config["vectordb"]["collection_name"],
            connection_args={"uri": milvus_uri},
        )
    
    def _setup_llm(self) -> HuggingFacePipeline:
        """
        Set up Hugging Face language model with quantization.
        
        :return: Configured language model pipeline
        """
        # BitsAndBytes configuration for model quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config['llm']['llm_path'],
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['llm']['llm_path'], 
            device_map="auto"
        )
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Define termination tokens
        terminators = [
            tokenizer.eos_token_id, 
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Create text generation pipeline
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
        
        return HuggingFacePipeline(pipeline=text_generation_pipeline)