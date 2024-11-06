import argparse
import os
import shutil
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.retrievers import BM25Retriever

from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import utils as chroma_utils

from typing import List, Tuple, Optional, BinaryIO, Union
from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig
from unstructured.staging.base import elements_from_json
from unstructured.chunking.title import chunk_by_title

import dotenv
import shutil
dotenv.load_dotenv("env.txt", override=True)
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
import pandas as pd
CHROMA_PATH = "chroma"
DATA ='./data'
DATA_PROCESSES ='./data_processed'

def process_file(file: str) -> str:
    """Process uploaded file"""
    if file is None:
        return "Please upload a file first."
    
    try:
        base_name = os.path.basename(file)
        shutil.copy(file,os.path.join(DATA,base_name))
        if base_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
            chunk_df = chunk_dataframe(df, chunk_size=1)
            # Convert chunks to the required Document format
            chunk_documents = []                
            metadata = {
                "source": base_name,
                "sheet_name": df.sheet_names if hasattr(df, 'sheet_names') else "Sheet1",
            }
            for chunk in chunk_df:
                chunk_documents.append(Document(page_content=chunk, metadata=metadata))
                chunk_documents = chroma_utils.filter_complex_metadata(chunk_documents)
        else:
            
            load_documents(DATA)
            chunk_documents = split_documents(DATA_PROCESSES)

        add_to_chroma(chunk_documents)
        os.remove(os.path.join(DATA,base_name))
        return "file processed successfully! You can now ask questions about its content."
        
    except Exception as e:
        return f"Error processing file: {str(e)} \n Please upload other file"
        

def chunk_dataframe(df, chunk_size):
    header = list(df.columns)  # Capture header as the title
    chunks = []

    # Iterate in chunks based on the specified chunk size
    for start_row in range(0, len(df), chunk_size):
        chunk = df.iloc[start_row:start_row + chunk_size]

        # Create a string for the first row's values
        if not chunk.empty:
            chunk_string = ', '.join(f"{header[i]}: {chunk.iloc[0, i]}" for i in range(len(header)))
            chunks.append(chunk_string)
        else:
            chunks.append("")  # Append an empty string if the chunk is empty

    return chunks

def load_documents(tmp_path: str):
    Pipeline.from_configs(
        context=ProcessorConfig(),
        indexer_config=LocalIndexerConfig(input_path=tmp_path),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        partitioner_config=PartitionerConfig(
            partition_by_api=True,
            api_key=os.environ.get("UNSTRUCTURED_API_KEY"),
            partition_endpoint=os.environ.get("UNSTRUCTURED_API_URL"),
            strategy="hi_res",

        ),
        uploader_config=LocalUploaderConfig(output_dir=DATA_PROCESSES)
    ).run()

def split_documents(output_processed_data: str):
    element =[]
    for filename in os.listdir(output_processed_data):
        filepath = os.path.join(output_processed_data, filename)
        element.extend(elements_from_json(filepath))
    chunk_elements = chunk_by_title(elements=element, max_characters=512, combine_text_under_n_chars=200)
    documents =[]
    for chunk_element in chunk_elements:
        metadata =chunk_element.metadata.to_dict()
        metadata["source"]= metadata["filename"]
        del metadata["languages"]
        print(chunk_element.text)
        documents.append(Document(page_content=chunk_element.text, metadata=metadata))
    documents = chroma_utils.filter_complex_metadata(documents)

    return documents

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()