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
        
        # Use reranker for deduplication and final selection
        final_results = rerank_results(query_text, all_matches, top_k=top_k)
        
        # Log warning if all scores are low
        if final_results and all(m['score'] < 0.4 for m in final_results):
            logger.warning(f"All matches have low scores (< 0.4) for query: '{query_text}'")
        
        return final_results
    
    except Exception as e:
        logger.error(f"Error querying Pinecone: {str(e)}")
        return []


def rerank_results(query_text: str, matches: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Rerank and deduplicate results.
    
    Current implementation: Title-based deduplication
    Future enhancement: Cross-encoder reranking for semantic relevance
    
    Trade-off: Cross-encoder would add 2-4s latency for 5-15% accuracy gain.
    For portfolio demo prioritizing fast UX, deduplication provides optimal balance.
    """
    if not matches:
        logger.warning(f"No valid results to rerank for query: '{query_text}'")
        return []
    
    # Deduplicate by title (case-insensitive)
    seen_titles = set()
    unique_matches = []
    
    for match in matches:
        title = match.get("title", "").lower().strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_matches.append(match)
    
    logger.info(f"Reranked {len(matches)} results to {len(unique_matches)} unique titles")
    
    return unique_matches[:top_k]


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
        # Track whether we use primary or fallback for metrics
        
        # Build a complete, well-structured prompt
        system_instructions = st.session_state.system_prompt
        
        # Format the prompt with clear sections
        full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_instructions}

RETRIEVED CONTENT YOU MUST USE:
{context}

CONVERSATION HISTORY:"""
        
        # Add conversation history
        for msg in history[-3:]:  # Last 3 exchanges to keep it concise
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                full_prompt += f"\nUser: {content}"
            elif role == "assistant":
                full_prompt += f"\nAssistant: {content}"
        
        # Add current message and start response
        full_prompt += f"\n\nUser: {message}\n\nAssistant: Well now, let me"
        
        # Try primary model first (best quality when it works)
        # Use chat completions API which works better with HF Pro
        try:
            logger.info("Attempting primary AI generation (Llama-3.1 via chat API)...")
            
            # Format as chat messages
            messages = [
                {
                    "role": "system",
                    "content": st.session_state.system_prompt
                },
                {
                    "role": "user", 
                    "content": f"""RETRIEVED CONTENT YOU MUST DISCUSS:
{context}

User Query: {message}

Remember: Discuss ALL {len(content_items)} titles with your Hollywood personality."""
                }
            ]
            
            # Create fresh client to avoid provider caching issues
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN")))
            
            # Use chat completions (works with Pro tier)
            response = client.chat_completion(
                messages=messages,
                model="meta-llama/Llama-3.1-8B-Instruct",
                max_tokens=st.session_state.get("max_tokens", 700),
                temperature=st.session_state.get("temperature", 0.7)
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Success! Use primary response
            logger.info("✅ Primary AI generation successful (Llama-3.1 chat API)")
            logger.info(f"AI Response preview: {response_text[:200]}...")  # Debug: see what we got
            return response_text
            
        except Exception as api_error:
            # Primary failed, use intelligent fallback
            logger.warning(f"Primary AI failed: {str(api_error)}, using intelligent fallback")
            raise  # Re-raise to trigger outer exception handler
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        logger.info("Using personality-infused fallback system")
        
        # Intelligent fallback with personality (maintains UX quality)
        if content_items:
            # Build fallback response with Hollywood personality
            response = "Well now, let me see what turned up in the archives, darling.\n\n"
            
            for i, item in enumerate(content_items, 1):
                response += f"**{i}. {item['title']}** ({item['year']}) - {item['genre']}\n"
                overview = item['overview'][:150] + "..." if len(item['overview']) > 150 else item['overview']
                response += f"   {overview}\n"
                
                # Add personality based on score
                score = item.get('score', 0)
                if score > 0.6:
                    response += f"   *Now we're talking! Strong match here, sweetheart.* (Score: {score:.3f})\n\n"
                elif score > 0.4:
                    response += f"   *A decent fit, I'd say.* (Score: {score:.3f})\n\n"
                else:
                    response += f"   *Bit of a stretch, but it's what the system found.* (Score: {score:.3f})\n\n"
            
            response += "\nThere you have it - the full picture of what my database turned up."
            logger.info("✅ Fallback response generated successfully")
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
    
    # Display retrieved content if available
    if st.session_state.retrieved_content:
        st.markdown("---")
        st.markdown("### 🎯 Last Retrieved")
        for i, item in enumerate(st.session_state.retrieved_content, 1):
            st.markdown(f"**{i}. {item['title']}** ({item['year']})")
            st.markdown(f"*Score: {item['score']:.3f}*")

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
    
    # Debug: Add cache clear button
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Main content area - chat messages
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
