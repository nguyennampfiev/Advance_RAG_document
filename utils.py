import multiprocessing
import os
import json
import yaml
import shutil
import logging
import tempfile
import time 
import pandas as pd
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.chunking import HybridChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from transformers import AutoTokenizer
from langchain.schema import Document  # Import the Document class

# Configure logging
logging.basicConfig(level=logging.INFO)

def detect_system_resources():
    """
    Detect available system resources for optimal processing
    
    Returns:
        Dictionary containing detected resources information
    """
    resources = {
        'cpu_cores': multiprocessing.cpu_count(),
        'gpu_available': False,
        'gpu_type': None
    }
    
    # Try to detect CUDA availability
    try:
        import torch
        resources['gpu_available'] = torch.cuda.is_available()
        if resources['gpu_available']:
            resources['gpu_type'] = 'cuda'
            resources['gpu_count'] = torch.cuda.device_count()
    except ImportError:
        pass
    
    # Try to detect MPS (Apple Metal) availability
    if not resources['gpu_available']:
        try:
            import torch
            resources['gpu_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if resources['gpu_available']:
                resources['gpu_type'] = 'mps'
                resources['gpu_count'] = 1  # MPS typically just reports as 1 device
        except ImportError:
            pass
    
    return resources

def load_config(config_path):
    """
    Load configuration from a JSON or YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

def get_accelerator_device(device_name):
    """
    Get the appropriate accelerator device based on config string
    
    Args:
        device_name: Name of the device ("cpu", "cuda", "mps", "auto")
        
    Returns:
        AcceleratorDevice enum value
    """
    device_map = {
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
        "auto": AcceleratorDevice.AUTO,
    }
    
    return device_map.get(device_name.lower(), AcceleratorDevice.CPU)

def split_pdf(input_pdf_path, output_folder, pages_per_chunk=10):
    """
    Split a PDF file into smaller chunks
    
    Args:
        input_pdf_path: Path to the input PDF file
        output_folder: Folder to save split PDF files
        pages_per_chunk: Number of pages per chunk
        
    Returns:
        List of paths to the split PDF files
    """
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_reader = PdfReader(input_pdf_path)
    total_pages = len(pdf_reader.pages)
    
    split_files = []
    
    for i in range(0, total_pages, pages_per_chunk):
        end_page = min(i + pages_per_chunk, total_pages)
        output_filename = os.path.join(output_folder, f"chunk_{i+1}_{end_page}.pdf")
        
        pdf_writer = PdfWriter()
        for page_num in range(i, end_page):
            pdf_writer.add_page(pdf_reader.pages[page_num])
            
        with open(output_filename, "wb") as out_file:
            pdf_writer.write(out_file)
            
        split_files.append(output_filename)
    
    return split_files

def process_pdf_chunk(pdf_path, pipeline_options):
    """
    Process a single PDF chunk
    
    Args:
        pdf_path: Path to the PDF chunk
        pipeline_options: Pipeline options for document conversion
        
    Returns:
        Conversion result
    """
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    
    # Convert the document
    conversion_result = converter.convert(pdf_path)
    
    # Get individual chunk timing if available
    if hasattr(conversion_result, 'timings') and conversion_result.timings:
        doc_conversion_secs = conversion_result.timings["pipeline_total"].times
    else:
        doc_conversion_secs = []
    
    return {
        'path': pdf_path,
        'result': conversion_result,
        'doc_conversion_secs': doc_conversion_secs
    }

def optimize_config_for_system(config):
    """
    Optimize configuration based on available system resources
    
    Args:
        config: User configuration dictionary
        
    Returns:
        Optimized configuration dictionary
    """
    # Create a copy to avoid modifying the original
    optimized_config = config.copy()
    
    # Detect system resources
    resources = detect_system_resources()
    
    # Initialize accelerator config if not present
    if 'accelerator' not in optimized_config:
        optimized_config['accelerator'] = {}
    
    # Configure device based on availability
    if 'device' not in optimized_config['accelerator'] or optimized_config['accelerator']['device'] == 'auto':
        if  resources['gpu_available']:
            optimized_config['accelerator']['device'] = resources['gpu_type']
            logging.info(f"Auto-detected {resources['gpu_type'].upper()} GPU - using for acceleration")
        else:
            optimized_config['accelerator']['device'] = 'cpu'
            logging.info("No GPU detected - using CPU for processing")
    
    # Optimize thread count if not specified
    if 'threads' not in optimized_config['accelerator'] or optimized_config['accelerator']['threads'] <= 0:
        # Use 75% of available cores by default
        optimized_config['accelerator']['threads'] = max(1, int(resources['cpu_cores'] * 0.5))
        logging.info(f"Auto-configured to use {optimized_config['accelerator']['threads']} threads based on system having {resources['cpu_cores']} cores")
    
    # Optimize worker count if not specified
    if 'max_workers' not in optimized_config or optimized_config['max_workers'] <= 0:
        # For CPU processing, use core count - 1 (minimum 1)
        # For GPU processing, set to 2x number of GPUs (gives some parallelism but doesn't overwhelm)
        if optimized_config['accelerator']['device'] == 'cpu':
            optimized_config['max_workers'] = max(1, resources['cpu_cores'] - 1)
        else:
            optimized_config['max_workers'] = max(1, min(resources.get('gpu_count', 1) * 2, resources['cpu_cores'] // 2))
        
        logging.info(f"Auto-configured to use {optimized_config['max_workers']} parallel workers")
    
    # Optimize chunk size based on hardware
    if 'pages_per_chunk' not in optimized_config or optimized_config['pages_per_chunk'] <= 0:
        # Smaller chunks for CPU (10 pages), larger for GPU (20 pages)
        #if optimized_config['accelerator']['device'] == 'cpu':
        #    optimized_config['pages_per_chunk'] = 10
        #else:
        optimized_config['pages_per_chunk'] = 20
        
        logging.info(f"Auto-configured to use {optimized_config['pages_per_chunk']} pages per chunk")
    
    return optimized_config
def batch_process_pdf(input_file, config):
    """
    Process a PDF file in batches using configuration
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        Dictionary with processing results and timing
    """
    # Optimize configuration based on system resources
    config = optimize_config_for_system(config)
    
    temp_dir = config.get('temp_dir', None)
    pages_per_chunk = config.get('pages_per_chunk', 10)
    max_workers = config.get('max_workers', 8)
    
    # Configure accelerator
    accelerator_config = config.get('accelerator', {})
    device_name = accelerator_config.get('device', 'cpu')
    num_threads = accelerator_config.get('threads', 8)
    
    accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=get_accelerator_device(device_name)
    )
    
    # Configure pipeline options
    pipeline_config = config.get('pipeline', {})
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = pipeline_config.get('do_ocr', True)
    pipeline_options.do_table_structure = pipeline_config.get('do_table_structure', True)
    pipeline_options.table_structure_options.do_cell_matching = pipeline_config.get('do_cell_matching', True)
    
    # Enable profiling if requested
    if config.get('profile_timings', True):
        settings.debug.profile_pipeline_timings = True
    
    # Create temporary directory if not specified
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp = True
    else:
        temp_dir = Path(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Measure overall start time
        overall_start_time = time.time()
        
        # Split PDF into chunks
        logging.info(f"Splitting PDF into chunks of {pages_per_chunk} pages each...")
        split_start_time = time.time()
        split_files = split_pdf(input_file, temp_dir, pages_per_chunk=pages_per_chunk)
        split_end_time = time.time()
        split_time = split_end_time - split_start_time
        logging.info(f"Split PDF into {len(split_files)} chunks in {split_time:.2f} seconds")
        
        # Process chunks in parallel
        logging.info(f"Processing chunks in parallel with {max_workers} workers...")
        logging.info(f"Using {device_name.upper()} with {num_threads} threads for acceleration")
        batch_start_time = time.time()
        
        processing_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_pdf_chunk, pdf_path, pipeline_options) 
                      for pdf_path in split_files]
            
            for future in futures:
                result = future.result()
                processing_results.append(result)
        
        batch_end_time = time.time()
        batch_process_time = batch_end_time - batch_start_time
        

        logging.info(f"initializing vector database...")
        vectorstore = create_milvus_db(config)
        for result in processing_results:
            doc = result['result'].document
            
            if doc:
                add_to_milvus(config, doc, vectorstore)
        logging.info(f"finished adding documents to vector database")
        
        # Overall completion time
        overall_end_time = time.time()
        overall_time = overall_end_time - overall_start_time
        
        # Calculate performance metrics
        pages_per_second = len(split_files) * pages_per_chunk / batch_process_time
        
        # # Collect timing information
        # results['timing'] = {
        #     'total': overall_time,
        #     'split': split_time,
        #     'batch_process': batch_process_time,
        #     'pages_per_second': pages_per_second
        # }
        
        # # Collect chunk information
        # for i, result in enumerate(processing_results):
        #     chunk_name = Path(result['path']).name
        #     chunk_info = {
        #         'name': chunk_name,
        #         'path': str(result['path'])
        #     }
            
        #     if result['doc_conversion_secs']:
        #         chunk_info['pipeline_time'] = sum(result['doc_conversion_secs'])
            
        #     results['chunks'].append(chunk_info)
        
        # Print timing information
        logging.info("\n--- Processing Summary ---")
        logging.info(f"Total elapsed time: {overall_time:.2f} seconds")
        logging.info(f"  - PDF splitting time: {split_time:.2f} seconds")
        logging.info(f"  - Batch processing time: {batch_process_time:.2f} seconds ({pages_per_second:.2f} pages/sec)")
        #logging.info(f"  - Results combining time: {combine_time:.2f} seconds")
        
    finally:
        # Clean up temporary directory if we created it
        if cleanup_temp:
            shutil.rmtree(temp_dir)
    
    #return results


def create_milvus_db(config: dict):
    """
    Create a new vector database with the specified embedding model.
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        The created Milvus vector store instance
    """
    embedding_config = config.get('embedding', {})
    embedding = embedding_config.get('embed_model_id', None)
    db_config = config.get('vectordb', {})
    db_path = db_config.get('path', './milvus')
    collection_name = db_config.get('collection_name', 'docling')
    milvus_uri = os.path.join(db_path, "docling.db")  # or set as needed
    #print(milvus_uri)
    #assert 1==2
    embedding_model = HuggingFaceEmbeddings(model_name=embedding)
    
    logging.info("Creating Milvus database...")
    # Initialize empty collection
    vectorstore = Milvus(
        embedding_model,
        collection_name=collection_name,
        connection_args={"uri": milvus_uri},
        index_params={"index_type": "FLAT"},
        drop_old=True,
        auto_id=True
    )
    logging.info("Milvus database created successfully")
    
    return vectorstore

def add_to_milvus(config: dict, splits: List, vectorstore: Any):
    """
    Add document splits to an existing Milvus database or create a new one.
    
    Args:
        config: Dictionary containing configuration parameters
        splits: The document splits to add to the database
        vectorstore: An existing Milvus vectorstore instance (optional)
        
    Returns:
        The updated Milvus vector store instance
    """
    embedding_config = config.get('embedding', {})
    db_config = config.get('vectordb', {})
    MILVUS_PATH = config.get('path', './milvus')
    milvus_uri = os.path.join(MILVUS_PATH, "docling.db")  # or set as needed
    EMBED_MODEL_ID = embedding_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

    collection_name = db_config.get('collection_name', 'docling')
    print("Adding documents to Milvus...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

    chunker = HybridChunker(
        tokenizer=tokenizer,  # instance or model name, defaults to "sentence-transformers/all-MiniLM-L6-v2"
        merge_peers=True,  # optional, defaults to True
    )
    chunk_iter = chunker.chunk(dl_doc=splits)
    chunks = list(chunk_iter)
    print(chunks[0])
    documents = [Document(page_content=doc.text) for doc in chunks]

    #splits =[doc.page_content for doc in splits]
    if vectorstore:
        # Add to existing vectorstore
        vectorstore.add_documents(documents)
    else:
        # Create new vectorstore with documents
        vectorstore = Milvus.from_documents(
            documents=splits,
            embedding=embedding_model,
            collection_name=collection_name,
            connection_args={"uri": milvus_uri},
            index_params={"index_type": "FLAT"},
            drop_old=True,
            auto_id=True
        )
    
    print("Finished adding documents")
    
    return vectorstore