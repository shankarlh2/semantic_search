import os
import json
import asyncio
import boto3
import faiss
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine

# --- CONFIG ---
DB_URL = os.getenv("DATABASE_URL")  # e.g., postgresql://user:pass@host/dbname
engine = create_engine(DB_URL)
sqs = boto3.client("sqs", region_name="ap-south-1")
queue_url = "https://sqs.ap-south-1.amazonaws.com/697957568578/article-processing-queue"

# --- APP ---
app = FastAPI()

# --- FAISS INDEX ---
dimension = 768
faiss_index = faiss.IndexFlatL2(dimension)
article_id_map = {}

# --- EMBEDDING MODEL ---
model = SentenceTransformer("all-distilroberta-v1")

# --- Request Models ---
class ArticleRequest(BaseModel):
    article_id: int
    title: str = None

class SearchRequest(BaseModel):
    title: str
    top_k: int = 5

# --- Initial DB Load ---
def load_db_articles_once():
    print("📥 Fetching articles from DB...")
    df = pd.read_sql("SELECT id, title FROM feed_results WHERE title IS NOT NULL", engine)

    if df.empty:
        print("⚠️ No articles found in DB.")
        return

    print(f"✅ Loaded {len(df)} articles from DB.")
    df = df.dropna(subset=["title"])
    titles = df["title"].tolist()
    ids = df["id"].tolist()

    embeddings = model.encode(titles, show_progress_bar=True).astype("float32")
    faiss_index.add(embeddings)

    for idx, article_id in enumerate(ids):
        article_id_map[idx] = article_id

    print("✅ FAISS index and ID map initialized.")

# --- Fetch from DB ---
def fetch_article(article_id: int):
    result = pd.read_sql(
        f"SELECT id, title FROM feed_results WHERE id = {article_id} AND title IS NOT NULL",
        engine
    )
    if result.empty:
        raise Exception("Article not found")
    return result.iloc[0]

# --- Add to FAISS ---
def add_to_faiss(article_id: int, title: str):
    if article_id in article_id_map.values():
        print(f"⚠️ Article ID {article_id} already indexed.")
        return

    embedding = model.encode([title])[0].astype("float32").reshape(1, -1)
    faiss_index.add(embedding)
    article_id_map[len(article_id_map)] = article_id
    print(f"✅ Article ID {article_id} added to FAISS.")

# --- Process article from SQS ---
async def process_article(article_id: int):
    try:
        article = fetch_article(article_id)
        add_to_faiss(article["id"], article["title"])
    except Exception as e:
        print(f"❌ Failed to process article {article_id}: {e}")

# --- SQS Polling ---
async def poll_sqs():
    print("📡 Starting SQS polling...")
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=10
            )
            if 'Messages' in response:
                for message in response['Messages']:
                    body = json.loads(message['Body'])
                    article_id = body['article_id']
                    print(f"📨 SQS received: article_id = {article_id}")
                    await process_article(article_id)
                    sqs.delete_message(
                        QueueUrl=queue_url,
                        ReceiptHandle=message['ReceiptHandle']
                    )
        except Exception as e:
            print(f"❌ Error polling SQS: {e}")
        await asyncio.sleep(1)

# --- Search API ---
@app.post("/search_similar_articles/")
async def search_similar_articles(req: SearchRequest):
    try:
        embedding = model.encode([req.title])[0].astype("float32").reshape(1, -1)
        distances, indices = faiss_index.search(embedding, req.top_k)

        results = []
        for i in range(req.top_k):
            faiss_idx = indices[0][i]
            article_id = article_id_map.get(faiss_idx)
            if article_id is not None:
                article = fetch_article(article_id)
                results.append({
                    "article_id": int(article["id"]),
                    "title": article["title"],
                    "score": float(1 - distances[0][i])
                })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

# --- Manual Embedding Endpoint ---
@app.post("/embed_article/")
async def embed_article(req: ArticleRequest):
    try:
        if req.title:
            add_to_faiss(req.article_id, req.title)
        else:
            article = fetch_article(req.article_id)
            add_to_faiss(article["id"], article["title"])
        return {"message": f"Embedded article {req.article_id}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Startup Hooks ---
@app.on_event("startup")
async def startup():
    print("🔄 Starting up...")
    load_db_articles_once()
    asyncio.create_task(poll_sqs())

# --- Healthcheck ---
@app.get("/")
def root():
    return {"status": "UP"}
