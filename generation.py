# app.py
import streamlit as st
import weaviate
import ollama
import numpy as np

# --- Config ---
WEAVIATE_CLASS_NAME = "MyLocalTSVFiles"
OLLAMA_MODEL = "llama2"  # or 'mistral', 'llama3', etc.
WEAVIATE_PORT = 8090
WEAVIATE_GRPC = 50052

# --- Embedding ---
def get_embedding(text):
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response['embedding']
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return [0.0] * 384

# --- Search ---
def search_weaviate(query, limit=3):
    client = weaviate.connect_to_local(
        host="localhost",
        port=WEAVIATE_PORT,
        grpc_port=WEAVIATE_GRPC
    )

    collection = client.collections.get(WEAVIATE_CLASS_NAME)
    embedding = get_embedding(query)
    
    response = collection.query.near_vector(
        near_vector=embedding,
        limit=limit
    )

    results = []
    for obj in response.objects:
        content = obj.properties['content']
        source = obj.properties['source_file']
        results.append((content, source))
    
    client.close()
    return results

# --- Ollama Completion ---
def get_response_from_ollama(context, query):
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""
    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, stream=False)
        return response['response'].strip()
    except Exception as e:
        return f"Ollama error: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="Ollama Search Chat", page_icon="💬")
st.title("💬 GPT-style Search with Ollama + Weaviate")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Ask a question...")

if user_query:
    with st.spinner("Searching..."):
        results = search_weaviate(user_query)
    
    if results:
        top_context = "\n\n---\n\n".join([r[0] for r in results])
        response = get_response_from_ollama(top_context, user_query)
    else:
        response = "No relevant results found."

    st.session_state.history.append((user_query, response))

# Display conversation
for user_msg, bot_msg in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
