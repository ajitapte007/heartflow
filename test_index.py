import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "hfn-nvc"
EMBEDDING_MODEL = "models/gemini-embedding-001"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    print("Error: detailed env vars missing")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def test_query(query):
    print(f"\nQuerying: '{query}'")
    embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )['embedding']

    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True
    )

    if not results['matches']:
        print("No matches found.")
        return

    print("Top results:")
    for match in results['matches']:
        source = match['metadata'].get('source', 'Unknown')
        score = match['score']
        print(f"- [Score: {score:.3f}] Source: {source}")

# Test queries for the different books
test_query("What is the central region?") # Spiritual Anatomy / generic
test_query("What is the role of the Master?") # Sahaj Marg
test_query("What is the nature of Truth?") # Truth Eternal
test_query("What are the four components of Nonviolent Communication?") # NVC
test_query("What does Reality at Dawn mean?") # Reality at Dawn
