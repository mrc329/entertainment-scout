"""
Entertainment Scout - Streamlit Version
RAG Chatbot with Pinecone + HuggingFace
"""

import os
import logging
from typing import Optional, List, Dict

import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from huggingface_hub import InferenceClient

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Entertainment Scout",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== INITIALIZATION ====================

@st.cache_resource
def initialize_services():
    """Initialize services - cached so it only runs once."""
    
    # Get API keys from Streamlit secrets or environment
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
    hf_token = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
    
    if not pinecone_api_key or not hf_token:
        st.error("⚠️ Missing API keys! Please set PINECONE_API_KEY and HF_TOKEN in secrets.")
        st.stop()
    
    logger.info("Initializing services...")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("tmdb")
    
    # Initialize SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize HuggingFace client
    hf_client = InferenceClient(api_key=hf_token)
    
    logger.info("✅ All services initialized")
    return index, model, hf_client


# Initialize services
try:
    index, model, hf_client = initialize_services()
except Exception as e:
    st.error(f"Failed to initialize: {str(e)}")
    st.stop()

# ==================== QUERY FUNCTIONS ====================

def query_pinecone(query_text: str, top_k: int = 5, min_score: float = 0.25) -> List[Dict]:
    """Query Pinecone and return results."""
    
    try:
        # 3x emphasis on query (matching indexing strategy)
        enriched_query = f"{query_text} {query_text} {query_text}"
        query_embedding = model.encode(enriched_query).tolist()
        
        all_matches = []
        
        # Query both namespaces
        for namespace in ["movies", "tv_shows"]:
            response = index.query(
                vector=query_embedding,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True
            )
            
            for match in response.matches:
                if match.score >= min_score:
                    all_matches.append({
                        "title": match.metadata.get("title", "Unknown"),
                        "type": match.metadata.get("type", "Unknown"),
                        "genre": match.metadata.get("genres", ""),
                        "year": match.metadata.get("release_year", ""),
                        "overview": match.metadata.get("overview", ""),
                        "score": match.score
                    })
        
        # Sort by score and deduplicate
        all_matches.sort(key=lambda x: x["score"], reverse=True)
        seen_titles = set()
        unique_matches = []
        
        for match in all_matches:
            title = match["title"].lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_matches.append(match)
        
        return unique_matches[:top_k]
    
    except Exception as e:
        logger.error(f"Error querying Pinecone: {str(e)}")
        return []


def generate_response(message: str, history: List[Dict], content_items: List[Dict]) -> str:
    """Generate chat response using HuggingFace."""
    
    if not content_items:
        return """I couldn't find anything matching that query in my database.

**Try asking about:**
- Game of Thrones, Breaking Bad, Friends, The Office
- La La Land, Coco, The Sound of Music, Wicked
- Or describe what you want: "Dark fantasy shows" or "Feel-good musicals"

**Note:** My database has 500+ titles, but coverage is limited. This is a portfolio demo showing RAG architecture with realistic constraints."""
    
    # Build context from retrieved content
    context = f"""You must recommend ONLY these {len(content_items)} titles. Do not mention any other shows or movies.

Available titles:"""
    
    for i, item in enumerate(content_items, 1):
        context += f"\n{i}. {item['title']} ({item['year']}) - {item['genre']}"
        context += f"\n   Plot: {item['overview'][:100]}..."
    
    context += "\n\nREMINDER: Only recommend from the list above."
    
    # Build messages
    messages = [
        {"role": "system", "content": context + "\n\n" + st.session_state.system_prompt}
    ]
    
    # Add history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    try:
        response = hf_client.chat_completion(
            messages,
            model="HuggingFaceH4/zephyr-7b-beta",
            max_tokens=st.session_state.get("max_tokens", 1024),
            temperature=st.session_state.get("temperature", 0.7),
            top_p=st.session_state.get("top_p", 0.9)
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"


# ==================== SESSION STATE ====================

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_content" not in st.session_state:
    st.session_state.retrieved_content = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """You're a chain-smoking movie and television connoisseur from Hollywood's golden age, a walking encyclopedia of cinema. You speak with a Mid-Atlantic accent, dripping with old-school charm and wit.

RESPONSE FORMAT:
1. Open with a witty introduction
2. For EACH recommendation: Title (Year) - Genre - Brief description
3. Include key cast where relevant
4. Close with a charming remark

CRITICAL: Only recommend titles from the provided list."""

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 1024

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "top_p" not in st.session_state:
    st.session_state.top_p = 0.9

# ==================== SIDEBAR ====================

with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("### Model Parameters")
    
    st.session_state.max_tokens = st.slider(
        "Response Length",
        min_value=256,
        max_value=2048,
        value=st.session_state.max_tokens,
        step=256
    )
    
    st.session_state.temperature = st.slider(
        "Temperature (creativity)",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1
    )
    
    st.session_state.top_p = st.slider(
        "Top P (diversity)",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.top_p,
        step=0.1
    )
    
    st.markdown("---")
    
    st.markdown("### System Prompt")
    st.session_state.system_prompt = st.text_area(
        "Customize the AI's personality:",
        value=st.session_state.system_prompt,
        height=200
    )
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retrieved_content = []
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 💡 Try These")
    example_queries = [
        "Shows like Game of Thrones",
        "Sitcoms like The Office",
        "Feel-good musicals",
        "Dark sci-fi shows"
    ]
    
    for query in example_queries:
        if st.button(query, use_container_width=True):
            # Add to messages and rerun
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

# ==================== MAIN CONTENT ====================

st.title("🎬 Entertainment Scout")
st.markdown("**Your personal entertainment connoisseur, powered by AI**")

# Info banner
with st.expander("📊 About This Demo", expanded=False):
    st.markdown("""
    - **Database:** 500+ curated titles spanning 90+ years
    - **Coverage:** MCU, Harry Potter, LOTR, Golden Age Hollywood, musicals, Oscar winners
    - **How it works:** Describe what you like, I'll recommend similar content
    - **Note:** Limited to my dataset - I'll let you know if I can't find matches
    
    *Portfolio project demonstrating RAG architecture with semantic search.*
    """)

# Display retrieved content if available
if st.session_state.retrieved_content:
    with st.sidebar:
        st.markdown("### 🎯 Retrieved Titles")
        for i, item in enumerate(st.session_state.retrieved_content, 1):
            st.markdown(f"**{i}. {item['title']}** ({item['year']})")
            st.markdown(f"*{item['genre']} | Score: {item['score']:.3f}*")
            st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Try: 'I like Game of Thrones' or 'Show me musicals'"):
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            # Query Pinecone
            retrieved = query_pinecone(prompt, top_k=5)
            st.session_state.retrieved_content = retrieved
        
        with st.spinner("Generating response..."):
            # Generate response
            response = generate_response(prompt, st.session_state.messages[:-1], retrieved)
        
        st.markdown(response)
    
    # Add assistant message to chat
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Entertainment Scout</strong> | RAG Architecture Demo</p>
    <p>Built with Pinecone, Sentence Transformers, HuggingFace & Streamlit</p>
</div>
""", unsafe_allow_html=True)
