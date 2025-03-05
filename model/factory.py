from typing import Dict, Any
from .llama_model import LlamaRAGModel

class ModelFactory:
    """
    Factory class for creating different types of RAG models.
    """
    
    @staticmethod
    def create_model(config: Dict[str, Any], context_window: int):
        """
        Create a model based on the specified type.
        
        :param config: Configuration dictionary
        :return: Instantiated model
        :raises ValueError: If an unsupported model type is specified
        """
        model_type = config["llm"]["model_type"]
        if model_type == "llama":
            return LlamaRAGModel(config, context_window=context_window)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")