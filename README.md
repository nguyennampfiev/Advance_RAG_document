## RAG Applications for Chatting with All Formats of Files (Offline processing)

### This repository demonstrates a powerful RAG (Retrieval-Augmented Generation) pipeline for interacting with various types of document files using LLMs (Large Language Models). The solution is built using Llama for LLM calls and docling for data processing, with an interactive demo showcased through Gradio.

### Key Features
- Offline processing: 
- LLM Integration: Utilizes the opensource llama and quantization with BitsAndBytesConfig.
- Data Processing: Leverages Docling for processing document, accelerate with multi-batch processing (handle ~500 pdf pages around 8 mins)
- Database: Milvus
- Interactive UI: Powered by Gradio for a user-friendly chat interface.
### Run
```
python run_gradio.py
```