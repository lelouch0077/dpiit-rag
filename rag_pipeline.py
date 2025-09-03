import os
import weaviate
import weaviate.classes as wvc
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. Configuration ---
# Configuration points to your local Docker instance
WEAVIATE_URL = "http://localhost:8080"
WEAVIATE_GRPC_URL = "localhost:50051"
WEAVIATE_CLASS_NAME = "MyLocalTextFiles_v4" # Using a new name for the v4 collection
SOURCE_FOLDER = "PS_4"
# CORRECTED: Use the special Docker DNS name to allow the Weaviate container
# to communicate with the Ollama server running on your host machine.
OLLAMA_API_URL = "http://host.docker.internal:11434"

# --- 2. Data Preparation (Chunking) ---
def chunk_file(file_path):
    """Reads a text file and splits it into smaller, manageable chunks."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150, # Increased overlap for better context
            length_function=len
        )
        return text_splitter.split_text(text)
    except Exception as e:
        print(f"Error reading or chunking {file_path}: {e}")
        return []

# --- 3. Weaviate v4 Setup and Indexing ---
def setup_and_index(client, class_name, folder_path):
    """
    Connects to Weaviate, sets up the schema using v4 syntax, and indexes files.
    """
    # Check if the collection already exists and delete it
    if client.collections.exists(class_name):
        print(f"Deleting existing collection: {class_name}")
        client.collections.delete(class_name)

    # Create a new collection using the v4 syntax
    print(f"Creating new collection: {class_name}")
    client.collections.create(
        name=class_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_ollama(
            model="nomic-embed-text",
            api_endpoint=OLLAMA_API_URL
        ),
        properties=[
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT)
        ]
    )
    print("Collection created successfully.")

    # Index the files
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' not found.")
        return

    print(f"Starting indexing for files in '{folder_path}'...")
    # Get the collection object
    my_collection = client.collections.get(class_name)

    # Use the batch manager for efficient indexing
    with my_collection.batch.dynamic() as batch:
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                chunks = chunk_file(file_path)
                
                if chunks:
                    print(f"  -> Adding {len(chunks)} chunks from {filename}")
                    for chunk in chunks:
                        data_object = {"content": chunk}
                        batch.add_object(properties=data_object)
    
    # Check for failed objects before confirming completion
    if len(my_collection.batch.failed_objects) > 0:
        print(f"ERROR: {len(my_collection.batch.failed_objects)} objects failed to index.")
        # Optionally, print the first few errors
        for failed in my_collection.batch.failed_objects[:5]:
            print(f" -> Failed object error: {failed.message}")
    else:
        print(f"Indexing complete. Total objects in collection: {len(my_collection)}")


# --- 4. Retrieval and Generation (v4 Syntax) ---
def search(client, class_name, query, limit=3):
    """Performs a semantic search in Weaviate using v4 syntax."""
    print(f"\n--- Performing search for query: '{query}' ---")
    
    my_collection = client.collections.get(class_name)
    
    response = my_collection.query.near_text(
        query=query,
        limit=limit
    )
    
    return response.objects

def generate_answer(query, retrieved_objects):
    """Generates a final answer using the local Ollama model."""
    print("--- Generating answer with Ollama (llama3) ---")
    
    # Extract the content from the retrieved objects
    context = "\n\n".join([obj.properties['content'] for obj in retrieved_objects])

    prompt = f"""
Based on the following context from internal documents, please provide a comprehensive answer to the user's question.

CONTEXT:
---
{context}
---

QUESTION:
{query}

ANSWER:
    """
    
    try:
        response = ollama.chat(
            model='llama3',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error during answer generation: {e}\nIs the Ollama server running?"

# --- Main Execution ---
if __name__ == "__main__":
    # Use a try-finally block to ensure the client connection is closed
    client = None # Initialize client to None
    try:
        # UPDATED: Connect to the local Weaviate instance using v4 client
        client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )
        print("Successfully connected to the local Weaviate instance (v4).")

        if not os.path.exists(SOURCE_FOLDER):
            os.makedirs(SOURCE_FOLDER)
            print(f"Created directory '{SOURCE_FOLDER}'. Please add your .txt files there.")

        # Combined setup and indexing step
        setup_and_index(client, WEAVIATE_CLASS_NAME, SOURCE_FOLDER)

        test_query = "What is the goal of the seventh framework programme?"
        results = search(client, WEAVIATE_CLASS_NAME, test_query)
        
        # Check if search returned any results
        if results:
            final_answer = generate_answer(test_query, results)
            print("\n\n--- FINAL ANSWER ---")
            print(final_answer)
        else:
            print("\n\nCould not retrieve any documents to answer the question.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your Docker container is running ('docker-compose up -d') and Ollama is running.")
    finally:
        # Ensure the client is closed if it was opened
        if client and client.is_connected():
            client.close()
            print("\nWeaviate client connection closed.")

