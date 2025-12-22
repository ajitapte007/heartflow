import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = "heartflow"
EMBEDDING_MODEL = "models/text-embedding-004"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY and PINECONE_API_KEY in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_path, start_page=1, end_page=None):
    """
    Extracts text from PDF file, page by page.
    """
    print(f"Reading PDF: {pdf_path}")
    read_mod = PdfReader(pdf_path)
    pages_content = []
    
    # Adjust for 0-based indexing
    start_index = start_page - 1
    end_index = end_page if end_page else len(read_mod.pages)
    
    pages_to_read = read_mod.pages[start_index:end_index]
    print(f"Reading pages {start_page} to {end_index} ({len(pages_to_read)} pages).")

    # Reading pages is typically fast enough to be sequential, but we show progress
    for i, page in enumerate(tqdm(pages_to_read, desc="Reading Pages", unit="page"), start=start_page):
        text = page.extract_text()
        if text:
            text = text.strip()
            pages_content.append({"page_number": i, "text": text})
            
    return pages_content

def chunk_text(pages_content, chunk_size=1000):
    chunks = []
    for page in pages_content:
        page_num = page["page_number"]
        paragraphs = page["text"].split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20: continue
            chunks.append({"page": page_num, "text": para})
    return chunks

def embed_batch(batch, batch_index):
    """
    Embeds a single batch of text.
    Returns (batch_index, embeddings) to keep order if needed (though we just map back).
    """
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type="retrieval_document",
            title="Book Content"
        )
        return batch_index, result['embedding']
    except Exception as e:
        print(f"Error embedding batch {batch_index}: {e}")
        return batch_index, None

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF into Pinecone using Gemini Embeddings (Parallel)")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--start_page", type=int, default=1, help="Page number to start processing from (1-indexed)")
    parser.add_argument("--end_page", type=int, default=None, help="Page number to stop processing at (inclusive)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the existing Pinecone index")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print("File not found.")
        return

    # 1. Read PDF
    pages = get_pdf_text(args.pdf_path, start_page=args.start_page, end_page=args.end_page)
    
    # 2. Chunk
    chunks = chunk_text(pages)
    print(f"Total Chunks: {len(chunks)}")
    if not chunks: return

    # 3. Embed (Parallel)
    texts = [c['text'] for c in chunks]
    batch_size = 100
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    vectors_map = {} # batch_index -> embeddings
    
    print("Generating Embeddings (Parallel)...")
    with ThreadPoolExecutor(max_workers=5) as executor: # Adjust workers based on rate limits
        futures = {executor.submit(embed_batch, batch, i): i for i, batch in enumerate(batches)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding Batches"):
            batch_idx, embeddings = future.result()
            if embeddings:
                vectors_map[batch_idx] = embeddings
            else:
                # Handle failure (retry logic could go here)
                pass

    # Flatten embeddings in correct order
    all_embeddings = []
    # We reconstruct the list based on batch index to match 'chunks' list order
    sorted_batch_indices = sorted(vectors_map.keys())
    for idx in sorted_batch_indices:
        all_embeddings.extend(vectors_map[idx])
        
    if len(all_embeddings) != len(chunks):
        print(f"Warning: Mismatch between chunks ({len(chunks)}) and embeddings ({len(all_embeddings)}). Some batches may have failed.")
        # Identify valid chunks? For now, we only proceed if we have a way to align.
        # Actually, since we keyed by batch index, we know which ones succeeded.
        # But if a batch completely failed, we lose those chunks. 
        # For this script, we'll just skip the failed batches' chunks.
        
        valid_chunks = []
        valid_embeddings = []
        for idx in sorted_batch_indices:
            start_chunk_idx = idx * batch_size
            end_chunk_idx = start_chunk_idx + len(vectors_map[idx]) # length of this batch's embeddings
            # (Should equal length of the text batch provided)
            
            # Re-slice the original chunk list to get the matching chunks
            batch_chunks = chunks[start_chunk_idx : start_chunk_idx + len(vectors_map[idx])]
            valid_chunks.extend(batch_chunks)
            valid_embeddings.extend(vectors_map[idx])
            
        chunks = valid_chunks
        all_embeddings = valid_embeddings

    # 4. Upsert (Parallel)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index {INDEX_NAME}...")
        try:
            pc.create_index(
                name=INDEX_NAME, 
                dimension=768, 
                metric="cosine", 
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
            )
            while not pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(1)
        except Exception as e:
            print(f"Error creating index: {e}")
            return
            
    index = pc.Index(INDEX_NAME)

    # Handle Overwrite
    if args.overwrite:
        print(f"Overwrite enabled: Clearing all vectors from index {INDEX_NAME}...")
        try:
            index.delete(delete_all=True)
            print("Index cleared.")
        except Exception as e:
            print(f"Error clearing index: {e}")

    
    print("Upserting to Pinecone...")
    
    # Prepare vector tuples
    vector_data = []
    for i, (chunk, vec) in enumerate(zip(chunks, all_embeddings)):
        _id = f"p{chunk['page']}_{i}"
        meta = {"text": chunk['text'], "page": chunk['page']}
        vector_data.append((_id, vec, meta))
        
    # Upsert batches
    upsert_batch_size = 100
    upsert_batches = [vector_data[i:i+upsert_batch_size] for i in range(0, len(vector_data), upsert_batch_size)]

    def upsert_batch_func(batch):
        index.upsert(vectors=batch)
        return len(batch)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(upsert_batch_func, b) for b in upsert_batches]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Upserting"):
            pass
            
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
