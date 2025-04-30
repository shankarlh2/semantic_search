import streamlit as st
import pandas as pd
import faiss
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# --- Ensure stopwords are available ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# --- Text preprocessing ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

# --- Load model, data, and index ---
@st.cache_resource
def load_assets():
    model = SentenceTransformer("all-distilroberta-v1")

    # Hardcoded CSV path and title column extraction
    df = pd.read_csv("data-1745943143329.csv")  # assumes it's in the working directory
    titles = df["title"].dropna().astype(str).tolist()

    cleaned_titles = [preprocess_text(title) for title in titles]
    embeddings = model.encode(cleaned_titles, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embeddings)

    # Build FAISS HNSW index
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16
    index.add(embeddings)

    return model, index, titles

model, index, titles = load_assets()

# --- Streamlit UI ---
st.title("🧠 Semantic News Search")
query = st.text_input("Enter your query:")

if query:
    cleaned_query = preprocess_text(query)
    query_embedding = model.encode([cleaned_query]).astype("float32")
    faiss.normalize_L2(query_embedding)
    k = 5
    distances, indices = index.search(query_embedding, k)

    st.markdown("### 🔍 Top Matches:")
    for i, idx in enumerate(indices[0]):
        st.write(f"{i+1}. {titles[idx]}  \n*Score: {1 - distances[0][i]:.4f}*")

