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
        
        # Log warning if all scores are low
        if unique_matches and all(m['score'] < 0.4 for m in unique_matches):
            logger.warning(f"All matches have low scores (< 0.4) for query: '{query_text}'")
        
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
    context = f"""MANDATORY: You MUST discuss ALL {len(content_items)} of these titles in your response.
Do not skip any. Do not pick favorites. Discuss EVERY SINGLE ONE.

Retrieved titles that you MUST include:"""
    
    for i, item in enumerate(content_items, 1):
        context += f"\n{i}. **{item['title']}** ({item['year']})"
        context += f"\n   - Genre: {item['genre']}"
        context += f"\n   - Plot: {item['overview'][:150]}..."
        context += f"\n   - Match Score: {item['score']:.3f}"
    
    context += f"\n\nREMINDER: Your response MUST mention all {len(content_items)} titles above. List them one by one."
    
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
        # Use HuggingFace Inference API with text generation
        # This works with standard read tokens
        
        # Format as a single prompt for text generation
        prompt = context + "\n\n" + st.session_state.system_prompt + "\n\n"
        
        # Add conversation history
        for msg in history[-3:]:  # Last 3 exchanges to keep it concise
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        # Add current message
        prompt += f"User: {message}\nAssistant:"
        
        # Use text generation with a free model
        response = hf_client.text_generation(
            prompt,
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=st.session_state.get("max_tokens", 512),
            temperature=st.session_state.get("temperature", 0.7),
            top_p=st.session_state.get("top_p", 0.9),
            return_full_text=False
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        
        # Fallback: Generate a simple response from retrieved content
        if content_items:
            response = f"Based on your query, I found these recommendations:\n\n"
            for i, item in enumerate(content_items[:3], 1):
                response += f"{i}. **{item['title']}** ({item['year']})\n"
                response += f"   Genre: {item['genre']}\n"
                response += f"   {item['overview'][:150]}...\n\n"
            return response
        else:
            return f"I encountered an error: {str(e)}\n\nPlease check your HuggingFace token permissions."


# ==================== SESSION STATE ====================

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieved_content" not in st.session_state:
    st.session_state.retrieved_content = []

if "example_query" not in st.session_state:
    st.session_state.example_query = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """You're a chain-smoking movie and television connoisseur from Hollywood's golden age, a walking encyclopedia of cinema spanning every era. You speak with a Mid-Atlantic accent, dripping with old-school charm and the right bit of snark.

CRITICAL RULES - Follow these EXACTLY:
1. You MUST discuss ALL titles from the provided list - no cherry-picking favorites
2. For EACH title, include: Title (Year) - Genre - Brief description with your characteristic wit
3. If a title doesn't match the query well, acknowledge it: "This one's a bit of a stretch, darling, but..."
4. Keep your personality in your DESCRIPTIONS, not in fake stage directions like "(Excitedly)"
5. No invented dialogue or actions - your charm comes through in HOW you describe the films, not fake emotions
6. Use the actual titles provided - do not invent or substitute others

FORMAT:
- Open with a characteristic quip about the query
- Discuss ALL retrieved titles, one by one, with your signature style
- For each: Title (Year) - Genre - Your witty take on why it fits (or doesn't)
- Close with a charming sign-off

Remember: Your personality shines through your word choices and observations, not fabricated emotions or cherry-picked recommendations."""

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
        if st.button(query, use_container_width=True, key=f"example_{query}"):
            # Clear any existing input
            st.session_state.example_query = query
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

# Handle example query button clicks
if "example_query" in st.session_state and st.session_state.example_query:
    prompt = st.session_state.example_query
    st.session_state.example_query = None  # Clear it
    
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
    st.rerun()

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
