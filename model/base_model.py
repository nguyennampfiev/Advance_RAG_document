import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

class AbstractRAGModel(ABC):
    """
    Enhanced RAG model with Gradio-compatible query method.
    """
    
    def __init__(self, config: Dict[str, Any], context_window: int = 5):
        """
        Initialize the RAG model with configuration and context management.
        
        :param config: Configuration dictionary
        :param context_window: Number of recent conversation turns to retain
        """
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize core components
        self.embedding = self._setup_embeddings()
        self.vector_store = self._setup_vector_store()
        self.retriever = self._setup_retriever()
        self.llm = self._setup_llm()
        self.qa_chain = self._setup_qa_chain()
        self.context_window = context_window
    
    def query(
        self, 
        query_text: str, 
        history: Optional[List[Tuple[str, str]]] = None,
        use_conversation_context: bool = True,
        include_document_context: bool = True
    ) -> str:
        """
        Enhanced query method compatible with Gradio interface.
        
        :param query_text: User's query
        :param history: Optional conversation history 
        :param use_conversation_context: Include recent conversation history
        :param include_document_context: Use document retrieval context
        :return: Model's response
        """
        # Prepare contexts
        contexts = []
        
        # Retrieve document context if enabled
        if include_document_context:
            doc_context_result = self.retriever.get_relevant_documents(query_text)
            document_context = "\n".join([doc.page_content for doc in doc_context_result])
            contexts.append(f"Document Context:\n{document_context}")
        
        # Add conversation context if enabled and history is provided
        if use_conversation_context and history:
            conversation_context = self._format_conversation_context(history)
            contexts.append(f"Conversation History:\n{conversation_context}")
        
        # Combine contexts
        full_context = "\n\n".join(contexts)
        
        # Prepare enhanced prompt
        enhanced_prompt = f"""
        Conversation Context and Question:
        {full_context}
        
        Question: {query_text}
        
        Please provide a comprehensive and contextually relevant answer.
        """
        
        # Invoke QA chain
        result = self.qa_chain.invoke({"query": enhanced_prompt})
        answer = result.get("result", "No result found.")
        
        return answer
    
    def _format_conversation_context(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history as context.
        
        :param history: Conversation history 
        :return: Formatted conversation context string
        """
        # Use only the last N turns based on context_window
        recent_history = history[-self.context_window * 2:]
        
        formatted_context = ""
        for turn in recent_history:
            formatted_context += f"{turn['role'].capitalize()}: {turn['content']}\n"
        
        return formatted_context

    def query_rag(self, query_text: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """
        Gradio-compatible query method that matches the expected interface.
        
        :param query_text: User's query
        :param history: Conversation history
        :return: Updated history and empty string (for Gradio)
        """
        # Get model's answer
        answer = self.query(query_text, history)
        
       # Update history - Gradio Chatbot expects a list of [user_message, assistant_message]
        updated_history = history + [
            {"role": "user", "content": query_text},
            {"role": "assistant", "content": answer}
        ]
        
        return updated_history
        

    # Existing abstract methods remain the same
    @abstractmethod
    def _setup_embeddings(self) -> HuggingFaceEmbeddings:
        """Set up embedding model."""
        pass
    
    @abstractmethod
    def _setup_vector_store(self) -> Milvus:
        """Set up vector store."""
        pass
    
    def _setup_retriever(self):
        """
        Set up retriever with default search parameters.
        Can be overridden in subclasses for custom retrieval.
        """
        return self.vector_store.as_retriever(search_kwargs={"k": 2})
    
    @abstractmethod
    def _setup_llm(self) -> HuggingFacePipeline:
        """Set up language model."""
        pass
    
    def _setup_qa_chain(self):
        """
        Set up QA chain with a default RAG prompt.
        Can be customized in subclasses.
        """
        prompt_rag_template = """
        <|start_header_id|>user<|end_header_id|>
        You are an assistant for answering questions using provided context. 
        You are given the extracted parts of a long document and a question. 
        Provide a conversational answer. If you don't know the answer, just say "I do not know." 
        Don't make up an answer.
        
        Question: {question}
        Context: {context}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_rag_template,
        )
        
        return RetrievalQA.from_chain_type(
            self.llm, 
            retriever=self.retriever, 
            chain_type_kwargs={"prompt": prompt}, 
            return_source_documents=False
        )