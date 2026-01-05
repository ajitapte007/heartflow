import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    print("Error: detailed env vars missing")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

def test_index(index_name, queries):
    print(f"\n\n=== Testing Index: {index_name} ===")
    if index_name not in pc.list_indexes().names():
        print(f"Index {index_name} not found!")
        return

    index = pc.Index(index_name)

    for query in queries:
        print(f"\nQuerying: '{query}'")
        embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )['embedding']

        results = index.query(
            vector=embedding,
            top_k=3,
            include_metadata=True
        )

        if not results['matches']:
            print("No matches found.")
            continue

        print("Top results:")
        for match in results['matches']:
            source = match['metadata'].get('source', 'Unknown')
            page = match['metadata'].get('page', '?')
            score = match['score']
            print(f"- [Score: {score:.3f}] {source} (Page {page})")

# Test Heartfulness Index
test_index("heartfulness", [
    "What is the central region?",             # Spiritual Anatomy
    "What is the role of the Master?",          # Sahaj Marg
    "What is the nature of Truth?"              # Truth Eternal
])

# Test NVC Index
test_index("nvc", [
    "What are the four components of Nonviolent Communication?",
    "How do we observe without evaluating?"
])
