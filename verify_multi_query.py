import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_HFN = "heartfulness"
INDEX_NVC = "nvc"
EMBEDDING_MODEL = "models/text-embedding-004"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    print("Error: detailed env vars missing")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_hfn = pc.Index(INDEX_HFN)
index_nvc = pc.Index(INDEX_NVC)

def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

def retrieve_context(query, top_k=3):
    print(f"\nQuerying: '{query}'")
    query_vector = get_embedding(query)
    
    def query_index(index_obj, name):
        try:
             results = index_obj.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
             return results['matches']
        except Exception as e:
            print(f"Error querying {name}: {e}")
            return []

    matches = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_hfn = executor.submit(query_index, index_hfn, "Heartfulness")
        future_nvc = executor.submit(query_index, index_nvc, "NVC")
        
        hfn_res = future_hfn.result()
        nvc_res = future_nvc.result()
        print(f"Heartfulness matches: {len(hfn_res)}")
        print(f"NVC matches: {len(nvc_res)}")
        
        matches.extend(hfn_res)
        matches.extend(nvc_res)

    contexts = []
    for match in matches:
        metadata = match['metadata']
        contexts.append({
            "source": metadata.get('source', 'Unknown'),
            "score": match['score']
        })
    
    return contexts

# Run test
results = retrieve_context("How does listening relate to the heart?")
print("\nCombined Results:")
for i, res in enumerate(results):
    print(f"{i+1}. {res['source']} (Score: {res['score']:.3f})")
