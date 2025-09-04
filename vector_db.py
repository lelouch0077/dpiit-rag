import os
import weaviate
from weaviate import classes as wvc
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import csv
import numpy as np

# --- Configuration ---
WEAVIATE_CLASS_NAME = "MyLocalTSVFiles"
SOURCE_FOLDER = r"C:\Users\Dell\Desktop\dpiit_ml_hackathon\tsv"

# --- Data Processing ---
def process_tsv_file(file_path):
    """Reads a TSV file and processes its content for embedding."""
    try:
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            
            for row in reader:
                text_content = " ".join([f"{key}: {value}" for key, value in row.items()])
                
                if len(text_content) > 500:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50,
                        length_function=len
                    )
                    split_chunks = text_splitter.split_text(text_content)
                    chunks.extend(split_chunks)
                else:
                    chunks.append(text_content)
        
        return chunks
    except Exception as e:
        print(f"Error reading or processing {file_path}: {e}")
        return []

# --- Manual Embedding Function ---
def get_embedding(text):
    """Get embedding from Ollama locally"""
    try:
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a dummy embedding if Ollama fails
        return [0.0] * 384  # nomic-embed-text has 384 dimensions
from tqdm import tqdm
# --- Weaviate Setup and Indexing ---
def setup_and_index(client, class_name, folder_path):
    
    """Sets up Weaviate and indexes files with manual embeddings."""
    if client.collections.exists(class_name):
        print(f"Deleting existing collection: {class_name}")
        client.collections.delete(class_name)

    print(f"Creating new collection: {class_name}")
    # Create collection WITHOUT vectorizer (we'll handle embeddings manually)
    collection = client.collections.create(
        name=class_name,
        vectorizer_config=None,  # No vectorizer - we handle embeddings
        properties=[
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="source_file", data_type=wvc.config.DataType.TEXT)
        ]
    )
    print("Collection created successfully.")

    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' not found.")
        return

    print(f"Starting indexing for files in '{folder_path}'...")
    my_collection = client.collections.get(class_name)

    with my_collection.batch.dynamic() as batch:
        for filename in (os.listdir(folder_path)):
            if filename.endswith(".tsv"):
                file_path = os.path.join(folder_path, filename)
                chunks = process_tsv_file(file_path)
                
                if chunks:
                    print(f"  -> Adding {len(chunks)} chunks from {filename}")
                    for chunk in tqdm(chunks):
                        # Generate embedding manually using Ollama
                        embedding = get_embedding(chunk)
                        data_object = {
                            "content": chunk,
                            "source_file": filename
                        }
                        # Add object with manual vector
                        batch.add_object(
                            properties=data_object,
                            vector=embedding
                        )
    
    print(f"Indexing complete. Total objects in collection: {len(my_collection)}")

# --- Search Function with Manual Embedding ---
def search(client, class_name, query, limit=3):
    """Performs semantic search using manual query embedding."""
    print(f"\n--- Performing search for query: '{query}' ---")
    
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    my_collection = client.collections.get(class_name)
    
    response = my_collection.query.near_vector(
        near_vector=query_embedding,
        limit=limit
    )
    
    return response.objects

# --- Main Execution ---
if __name__ == "__main__":
    client = None
    try:
        # Connect to Docker Weaviate on different ports (to avoid port 8080 conflict)
        client = weaviate.connect_to_local(
            host="localhost",
            port=8090,      # Use port 8090 instead of 8080
            grpc_port=50052 # Use port 50052 instead of 50051
        )
        print("Successfully connected to Weaviate Docker instance on port 8090.")

        # Check if Ollama is running
        try:
            ollama.list()
            print("Ollama connection successful.")
        except:
            print("Warning: Ollama may not be running. Using dummy embeddings.")

        if not os.path.exists(SOURCE_FOLDER):
            os.makedirs(SOURCE_FOLDER)
            print(f"Created directory '{SOURCE_FOLDER}'. Please add your .tsv files there.")
            exit()

        setup_and_index(client, WEAVIATE_CLASS_NAME, SOURCE_FOLDER)
        print("Embedding process completed successfully!")

        # Test search
        test_query = "What information can you find about the data?"
        results = search(client, WEAVIATE_CLASS_NAME, test_query)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result.properties['content'][:100]}...")
        else:
            print("\nNo results found.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure Docker is running and Weaviate container is started with:")
        print("docker-compose up -d")
    finally:
        if client and client.is_connected():
            client.close()
            print("Weaviate client connection closed.")