import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import os
import pandas as pd
import json
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

# Initialize Google Sheets Connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Helper to create a unique hash for the question
def get_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

INDEX_HFN = "heartfulness"
INDEX_NVC = "nvc"
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash-lite"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("Please set GOOGLE_API_KEY and PINECONE_API_KEY in .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_hfn = pc.Index(INDEX_HFN)
index_nvc = pc.Index(INDEX_NVC)

st.set_page_config(page_title="HeartFlow", page_icon="resources/hfn_favicon_white.png", layout="wide")

def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

def retrieve_context(query, top_k=3):
    query_vector = get_embedding(query)
    
    if not query_vector:
        st.error("Failed to generate embedding for the query.")
        return []

    if not isinstance(query_vector, list):
        query_vector = list(query_vector)

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

    # Query both indices in parallel
    matches = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_hfn = executor.submit(query_index, index_hfn, "Heartfulness")
        future_nvc = executor.submit(query_index, index_nvc, "NVC")
        
        matches.extend(future_hfn.result())
        matches.extend(future_nvc.result())

    # Format contexts
    contexts = []
    for match in matches:
        metadata = match['metadata']
        contexts.append({
            "text": metadata['text'],
            "page": int(metadata['page']),
            "source": metadata.get('source', 'Unknown'),
            "score": match['score']
        })
    
    return contexts



def generate_direct_answer(query):
    prompt = f"""You are a helpful assistant.
Question: {query}

Answer:"""
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt, stream=True)
    return response

# UI
st.title("HeartFlow")
st.markdown("*Explore the wisdom in the Heartfulness literature, one question at a time.*")

st.info("💡 **Tip:** You can either browse **suggested questions** by topic using the dropdowns below, OR simply **type your own question** directly in the text box.")

def load_questions(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return {}

qa_data = load_questions("resources/questions.json")

# 1. Category Selection
categories = list(qa_data.keys()) if isinstance(qa_data, dict) else []
selected_category = st.selectbox(
    "1. Filter by Topic:", 
    categories, 
    index=None, 
    placeholder="Select a topic..."
)

# 2. Question Selection
questions = qa_data.get(selected_category, []) if isinstance(qa_data, dict) and selected_category else []
selection = st.selectbox(
    "2. Pick a question related to the topic:", 
    questions, 
    index=None, 
    placeholder="Select a question..."
)

default_text = selection if selection else ""
with st.form(key="query_form", clear_on_submit=False):
    query = st.text_input(
        "OR, ask your own question:", 
        value=default_text,
        placeholder="Type your question here..."
    )
    
    submit_button = st.form_submit_button(label="Submit")

async def async_retrieve_context(query, top_k=5):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        # Pass top_k=3 (per index) to get 6 total
        return await loop.run_in_executor(pool, retrieve_context, query, 3)

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
                st.markdown(f"**Source**: {c.get('source', 'Unknown')} | **Page {int(c['page'])}** (Score: {c['score']:.4f})")
                st.caption(c['text'])
                st.divider()

        context_str = "\n\n".join([f"Source: {c.get('source', 'Unknown')}, Page {int(c['page'])}: {c['text']}" for c in contexts])
        prompt = f"""You are a Heartfulness guide, Mr. Kamlesh Patel, affectionately known as Daaji. You are an expert in Sahaj Marg (Heartfulness) and also integrate the principles of Nonviolent Communication (NVC) to help seekers bridge their inner spiritual state with their outer relationships.

    You will be provided with context from two indices:
    1. [Heartfulness Content]: Focusing on meditation, cleaning, and spiritual evolution.
    2. [NVC Content]: Focusing on needs, feelings, and empathetic communication.

    Instructions:
    - Tone & Voice: Maintain a professional, wise, and gentle tone. Be empathetic but avoid overly familiar or patronizing forms of address like "My dear child." Your guidance should be simple, practical, and deeply rooted in the heart.
    - Structure: Adapt the structure to the nature of the query:
        - **For Complex/Life Queries** (e.g., relationships, stress, ego): Usage of the split structure (**"Inner Work"** and **"Outer Expression"**) is highly recommended to provide a complete answer.
        - **For Technical/Specific Queries** (e.g., "What is cleaning?", "Where is Point A?"): Focus directly on addressing the core question without forcing a division or unnecessary NVC content.
        - **Summary**: For complex answers, end with a **"Summary"** blending the perspectives.
    - Integration: Only integrate NVC concepts if they genuinely add value to the answer. Do not shoe-horn them into purely technical spiritual questions.
    - Narrative: Include anecdotes or stories from the provided context whenever possible. However, ALWAYS narrate them in the third person (e.g., "There was once a student who...", "One master observed...") even if the source uses "I". Do not use the first person ("I") for anecdotes to avoid confusion between different source authors.
    - Perspective: Always keep the conversation between "you" (Daaji) and "the user." Do not say "the context says" or "the text mentions." Speak as if this wisdom is your own.
    - Relevance: If the query is unrelated to either Heartfulness or compassionate living/NVC, politely decline to answer.
    - Citation: Cite your sources clearly using the name of the book and page number, formatted like (Spiritual Anatomy, Page 123).

    Example Integration: If a user asks about anger, use Heartfulness context to explain cleaning the "complexities" of the heart, and use NVC to explain how to identify the "unmet need" behind that anger.
    
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
    st.session_state["current_query"] = query
    asyncio.run(main_loop(query))
elif submit_button and not query:
    st.warning("Please enter a question before submitting.")

# Feedback form
if "current_query" in st.session_state:
    st.divider()
    st.subheader("Help us improve HeartFlow")

    with st.form(key="rating_form"):
        st.write(f"**Question:** {st.session_state.current_query}")
        
        rating = st.radio(
            "**1. Which response felt more helpful for your spiritual practice?**",
            options=[
                "Guided answer", 
                "General answer",
                "Both were helpful",
                "Neither was helpful"
            ],
            index=None,
            help="Your feedback helps refine the HeartFlow guidance logic."
        )

        reasons = st.multiselect(
            "**2. What could be improved? (Select all that apply)**",
            [
                "Irrelevant", 
                "Hard to understand", 
                "Not enough detail", 
                "Too verbose", 
                "Inaccurate references"
            ]
        )
    
        other_text = st.text_area("**3. Additional comments:**", placeholder="Share your thoughts here...")
        selected_reasons = ", ".join(reasons)
        
        feedback_submitted = st.form_submit_button("Submit Feedback")

        if feedback_submitted:
            if rating:  
                new_row = pd.DataFrame([{
                    "hashkey": get_hash(st.session_state.current_query),
                    "question": st.session_state.current_query,
                    "rating": rating,
                    "reasons": selected_reasons,
                    "comment": other_text
                }])
                try:
                    sheet_id = st.secrets["connections"]["gsheets"]["spreadsheet"]
                    existing_data = conn.read(worksheet="pilot")
                    updated_df = pd.concat([existing_data, new_row], ignore_index=True)
                    conn.update(worksheet="pilot", data=updated_df)
                    st.success("Thank you! Your feedback has been recorded.")
                except Exception as e:
                    st.error(f"Could not save to Google Sheets: {e}")
            else:
                st.warning("Please select an option before submitting.")
