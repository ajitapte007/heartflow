import os
import time
import argparse
import re
from uuid import uuid4
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
EMBEDDING_MODEL = "models/gemini-embedding-001"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY and PINECONE_API_KEY in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_path, start_page=1, end_page=None, title="PDF"):
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
    for i, page in enumerate(tqdm(pages_to_read, desc=f"Reading {title}", unit="page"), start=start_page):
        text = page.extract_text()
        if text:
            text = text.strip()
            pages_content.append({"page_number": i, "text": text})
            
    return pages_content

def get_markdown_text(md_path, start_page=1, end_page=None, title="Markdown"):
    """
    Extracts text from Markdown file, splitting by '## Page N'.
    """
    print(f"Reading Markdown: {md_path}")
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading markdown file: {e}")
        return []

    # Split by '## Page <number>'
    # resulting list will be [preamble, page_num_1, text_1, page_num_2, text_2, ...]
    # or [preamble, text_1] if no page numbers found (fallback)
    
    pattern = r'(## Page \d+\n)'
    parts = re.split(pattern, content)
    
    pages_content = []
    
    # Check if we have page markers. If not, treat whole file as one page (or page 1)
    if len(parts) < 2:
        print("No '## Page N' markers found. Treating as single page.")
        return [{"page_number": 1, "text": content.strip()}]
    
    # Iterate through parts. 
    # parts[0] is text before first page marker (often front matter).
    # parts[1] is marker (e.g. "## Page 1\n"), parts[2] is text for that page, etc.
    
    # We can handle front matter as page 0 or skip it. Let's skip valid pages only.
    
    current_page_num = 0
    
    # Verify if parts[0] has content? for now skip
    
    for i in range(1, len(parts), 2):
        marker = parts[i]
        text = parts[i+1] if i+1 < len(parts) else ""
        
        # Extract number from marker
        m = re.search(r'## Page (\d+)', marker)
        if m:
            page_num = int(m.group(1))
            
            # Filter ranges
            if page_num < start_page:
                continue
            if end_page and page_num > end_page:
                break # sorted assumption? or just continue
                
            pages_content.append({
                "page_number": page_num,
                "text": text.strip()
            })
            
    print(f"Extracted {len(pages_content)} pages from Markdown.")
    return pages_content

def chunk_text(pages, chunk_size=1000, overlap=100):
    chunks = []
    for page in pages:
        page_num = int(page["page_number"])
        text = page["text"]
        paragraphs = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50: continue
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

def process_file(file_path, title, index, args, start_page=1, end_page=None):
    """
    Full pipeline for a single file (PDF or Markdown).
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\n--- Processing: {title} ({file_path}) ---")

    # 1. Read File
    if file_path.lower().endswith('.md'):
        pages = get_markdown_text(file_path, start_page=start_page, end_page=end_page, title=title)
    else:
        # Default to PDF
        pages = get_pdf_text(file_path, start_page=start_page, end_page=end_page, title=title)
    
    # 2. Chunk
    chunks = chunk_text(pages, chunk_size=args.chunk_size)
    print(f"Total Chunks: {len(chunks)}")
    if not chunks: return

    # 3. Embed (Parallel)
    texts = [c['text'] for c in chunks]
    batch_size = 100
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    vectors_map = {} # batch_index -> embeddings
    
    print("Generating Embeddings (Parallel)...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(embed_batch, batch, i): i for i, batch in enumerate(batches)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Embedding {title}"):
            batch_idx, embeddings = future.result()
            if embeddings:
                vectors_map[batch_idx] = embeddings
            else:
                pass

    all_embeddings = []
    sorted_batch_indices = sorted(vectors_map.keys())
    
    if len(sorted_batch_indices) != len(batches):
         print(f"Warning: Only {len(sorted_batch_indices)}/{len(batches)} batches succeeded.")
         
    valid_chunks = []
    valid_embeddings = []
    
    for idx in sorted_batch_indices:
        start_chunk_idx = idx * batch_size
        actual_batch_len = len(vectors_map[idx])
        
        batch_chunks = chunks[start_chunk_idx : start_chunk_idx + actual_batch_len]
        valid_chunks.extend(batch_chunks)
        valid_embeddings.extend(vectors_map[idx])
            
    chunks = valid_chunks
    all_embeddings = valid_embeddings

    # 4. Upsert
    print(f"Upserting {len(chunks)} vectors to Pinecone...")
    
    vector_data = []
    book_slug = title.lower().replace(" ", "_")
    
    for i, (chunk, vec) in enumerate(zip(chunks, all_embeddings)):
        _id = f"{book_slug}_p{chunk['page']}_{i}"
        meta = {"text": chunk['text'], "page": chunk['page'], "source": title}
        vector_data.append((_id, vec, meta))
        
    upsert_batch_size = 100
    upsert_batches = [vector_data[i:i+upsert_batch_size] for i in range(0, len(vector_data), upsert_batch_size)]

    def upsert_batch_func(batch):
        index.upsert(vectors=batch)
        return len(batch)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(upsert_batch_func, b) for b in upsert_batches]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Upserting {title}"):
            pass
            
    print(f"Completed: {title}")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF(s) into Pinecone using Gemini Embeddings")
    
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file (list of dicts with: pdf_path, start_page, end_page, title)")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for text processing")
    parser.add_argument("--overwrite", action="store_true", help="Clear the ENTIRE index before upserting (Caution!)")
    parser.add_argument("--index_name", type=str, default="hfn_nvc", help="Pinecone index name")
    args = parser.parse_args()
    
    # Setup Pinecone Index ONCE
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    index_name = args.index_name
    
    if index_name not in pc.list_indexes().names():
        print(f"Creating index {index_name}...")
        try:
            pc.create_index(
                name=index_name, 
                dimension=3072, 
                metric="cosine", 
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        except Exception as e:
            print(f"Error creating index: {e}")
            return
            
    index = pc.Index(index_name)

    # Handle Overwrite ONCE
    if args.overwrite:
        print(f"Overwrite enabled: Clearing all vectors from index {index_name}...")
        try:
            index.delete(delete_all=True)
            print("Index cleared.")
        except Exception as e:
            print(f"Error clearing index: {e}")

    # Determine Processing List
    tasks = []
    import json
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
    try:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            # Expecting a list of dicts
            if isinstance(config_data, list):
                tasks = config_data
            else:
                print("Config file must contain a JSON list of objects.")
                return
    except Exception as e:
        print(f"Error reading config: {e}")
        return

    for task in tasks:
        # Support "pdf_path" (legacy) or "file_path" or generic "path"
        file_path = task.get("pdf_path") or task.get("file_path") or task.get("path")
        if not file_path: continue
        
        title = task.get("title")
        if not title:
             title = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title()
        
        start = task.get("start_page", 1)
        end = task.get("end_page", None)
        
        process_file(file_path, title, index, args, start_page=start, end_page=end)

    print("\nAll Ingestion Tasks Complete!")

if __name__ == "__main__":
    main()
