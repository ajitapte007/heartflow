import asyncio
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import os
import json
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

INDEX_NAME = "heartflow"
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash-lite"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("Please set GOOGLE_API_KEY and PINECONE_API_KEY in .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

st.set_page_config(page_title="HeartFlow", page_icon="resources/hfn_favicon_white.png", layout="wide")

def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

def retrieve_context(query, top_k=5):
    query_vector = get_embedding(query)
    
    if not query_vector:
        st.error("Failed to generate embedding for the query.")
        return []

    if not isinstance(query_vector, list):
        query_vector = list(query_vector)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    contexts = []
    for match in results['matches']:
        metadata = match['metadata']
        contexts.append({
            "text": metadata['text'],
            "page": metadata['page'],
            "score": match['score']
        })
    return contexts

def generate_answer(query, contexts):
    context_str = "\n\n".join([f"Page {c['page']}: {c['text']}" for c in contexts])
    
    prompt = f"""You are a Heartfulness guide, Mr. Kamlesh Patel affectionately known as Daaji.
    
    You will be given a user's query and some Heartfulness content relevant to the query.
    Instructions:
    - Maintain a tone that is authoritative yet gentle and empathetic. The answer should sound simple and practical.
    - Include anecdotes and stories from the context wherever you can and if relevant to the query, to make the answer more relatable and engaging.
    - Prefer to use the context provided to answer the question. Include additional information only as support if needed.
    - Always keep it between you and the user. Never refer to "the context" or "the author" in the third person.
    - If the answer cannot be derived from the context, say so.
    - Politely refuse to answer any questions that are not related to Heartfulness.
    - Cite the page numbers in your answer.
    
    Context:
    {context_str}

    Question: {query}

    Answer:"""

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt, stream=True)
    return response

def generate_direct_answer(query):
    prompt = f"""You are a helpful assistant.
Question: {query}

Answer:"""
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt, stream=True)
    return response

# UI
st.title("HeartFlow")

def load_questions(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return []

options = load_questions("resources/questions.json")
selection = st.selectbox("Pick a suggested question:", options)
default_text = ""
if selection != options[0] and selection != options[-1]:
    default_text = selection
with st.form(key="query_form", clear_on_submit=False):
    query = st.text_input(
        "OR, ask your own:", 
        value=default_text,
        placeholder="Type your question here..."
    )
    
    submit_button = st.form_submit_button(label="Submit")

async def async_retrieve_context(query, top_k=5):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, retrieve_context, query, top_k)

async def async_generate_rag(query, container, context_container):
    try:
        with context_container:
            with st.spinner("Retrieving content..."):
                contexts = await async_retrieve_context(query)
        
        if not contexts:
            context_container.warning("No relevant content found.")
            return

        with context_container.expander("View referenced content"):
            for c in contexts:
                st.markdown(f"**Page {int(c['page'])}** (Score: {c['score']:.4f})")
                st.caption(c['text'])
                st.divider()

        context_str = "\n\n".join([f"Page {int(c['page'])}: {c['text']}" for c in contexts])
        prompt = f"""You are a helpful assistant answering questions about the Heartfulness meditation system.
    
    You will be given a question and some relevant content related to Heartfulness.
    If the answer is not in the context, say so.
    Cite the page numbers in your answer.
    The tone should be friendly and helpful and should sound like native knowledge, not referring to "the context" or "the author".
    
    Context:
    {context_str}

    Question: {query}

    Answer:"""
        
        import google.generativeai as genai # Ensure genai is imported
        model = genai.GenerativeModel(GENERATION_MODEL)
        
        with container:
            with st.spinner("Generating guided answer..."):
                response_stream = await model.generate_content_async(prompt, stream=True)
                
                full_text = ""
                async for chunk in response_stream:
                    if chunk.text:
                        full_text += chunk.text
                        container.markdown(full_text + "▌")
        container.markdown(full_text)
        
    except Exception as e:
        container.error(f"Error in HFN answer: {e}")

async def async_generate_direct(query, container):
    try:
        import google.generativeai as genai
        prompt = f"""You are a helpful assistant.
Question: {query}

Answer:"""
        model = genai.GenerativeModel(GENERATION_MODEL)
        
        with container:
            with st.spinner("Generating general answer..."):
                response_stream = await model.generate_content_async(prompt, stream=True)
                
                full_text = ""
                async for chunk in response_stream:
                    if chunk.text:
                        full_text += chunk.text
                        container.markdown(full_text + "▌")
        container.markdown(full_text)
    except Exception as e:
        container.error(f"Error in Gemini answer: {e}")

async def main_loop(query):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Guided answer")
        rag_context_spot = st.empty() 
        rag_answer_spot = st.empty()
        
    with col2:
        st.subheader("General answer")
        direct_answer_spot = st.empty()

    await asyncio.gather(
        async_generate_rag(query, rag_answer_spot, rag_context_spot),
        async_generate_direct(query, direct_answer_spot)
    )

if submit_button and query:
    asyncio.run(main_loop(query))
elif submit_button and not query:
    st.warning("Please enter a question before submitting.")
