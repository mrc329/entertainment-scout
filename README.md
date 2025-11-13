# 🎬 Entertainment Scout

A conversational AI recommendation engine powered by RAG (Retrieval-Augmented Generation) architecture. Query a database of 500+ movies and TV shows using natural language and get personalized recommendations.

## Features

- **Semantic Search**: Vector-based search using Sentence Transformers
- **RAG Architecture**: Retrieves relevant content before generating responses
- **Conversational AI**: Natural dialogue powered by HuggingFace LLMs
- **Smart Fallbacks**: Graceful handling when queries fall outside dataset coverage

## Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: Pinecone
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: HuggingFace Inference API (Zephyr-7B)

---

## 🚀 Deployment Instructions

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at [streamlit.io](https://streamlit.io))
- Pinecone API key
- HuggingFace token

---

## Step 1: Push to GitHub

### 1.1 Create a New Repository

```bash
# Navigate to your project directory
cd /path/to/your/project

# Initialize git (if not already initialized)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Entertainment Scout"
```

### 1.2 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `entertainment-scout` (or your choice)
3. Description: "AI-powered entertainment recommendation chatbot"
4. Choose **Public** or **Private**
5. Do **NOT** initialize with README (you already have one)
6. Click **Create repository**

### 1.3 Push to GitHub

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/entertainment-scout.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Verify**: Visit your repository URL to confirm files are uploaded.

---

## Step 2: Deploy to Streamlit Cloud

### 2.1 Sign Up for Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Click **Sign in** (use your GitHub account)
3. Authorize Streamlit to access your GitHub

### 2.2 Create New App

1. Click **New app** button
2. **Repository**: Select `YOUR_USERNAME/entertainment-scout`
3. **Branch**: `main`
4. **Main file path**: `streamlit_app.py`
5. **App URL**: Choose your custom subdomain (e.g., `entertainment-scout`)

### 2.3 Configure Secrets

Before deploying, add your API keys:

1. Click **Advanced settings**
2. In the **Secrets** section, paste:

```toml
PINECONE_API_KEY = "your-pinecone-api-key-here"
HF_TOKEN = "your-huggingface-token-here"
```

3. Click **Save**
4. Click **Deploy**

### 2.4 Wait for Deployment

- Streamlit will install dependencies (2-3 minutes)
- Watch the logs for any errors
- Once complete, your app will be live at: `https://entertainment-scout.streamlit.app`

---

## 🔧 Local Development

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/entertainment-scout.git
cd entertainment-scout

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Add Local Secrets

Create `.streamlit/secrets.toml`:

```toml
PINECONE_API_KEY = "your-pinecone-api-key"
HF_TOKEN = "your-huggingface-token"
```

### Run Locally

```bash
streamlit run streamlit_app.py
```

App will open at `http://localhost:8501`

---

## 📝 Project Structure

```
entertainment-scout/
├── streamlit_app.py          # Main application
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # This file
└── .streamlit/
    ├── config.toml           # Streamlit configuration
    └── secrets.toml.template # Template for secrets
```

---

## 🎯 Usage

### Example Queries

- "Shows like Game of Thrones"
- "Sitcoms like The Office"
- "I want a dark fantasy series"
- "Recommend feel-good musicals"
- "Something like Breaking Bad"

### Customization

**Adjust AI Personality**: Use the sidebar to edit the system prompt

**Model Parameters**:
- **Response Length**: Control output verbosity
- **Temperature**: Adjust creativity (0.1 = focused, 1.0 = creative)
- **Top P**: Control diversity in word selection

---

## 🔑 Getting API Keys

### Pinecone API Key

1. Sign up at https://www.pinecone.io
2. Create a new project
3. Go to **API Keys** tab
4. Copy your API key

### HuggingFace Token

1. Sign up at https://huggingface.co
2. Go to Settings → Access Tokens
3. Create new token (read access sufficient)
4. Copy token

---

## 🐛 Troubleshooting

### "Missing API keys" error
- Verify secrets are added in Streamlit Cloud dashboard
- Check spelling: `PINECONE_API_KEY` and `HF_TOKEN`

### App won't load
- Check Streamlit logs for errors
- Verify `requirements.txt` dependencies
- Ensure main file is `streamlit_app.py`

### Slow responses
- Free tier HuggingFace API can be slow
- Consider upgrading to paid tier for faster inference

### "No relevant content found"
- Database has 500 titles - some queries won't match
- Try asking about: Game of Thrones, Breaking Bad, Friends, The Office, musicals

---

## 📊 Dataset Coverage

**Included**:
- Marvel Cinematic Universe (complete)
- Harry Potter series
- Lord of the Rings trilogy
- Popular TV shows (Friends, The Office, Breaking Bad, Game of Thrones)
- Classic films (Casablanca, The Godfather, etc.)
- Modern musicals (La La Land, Wicked, etc.)

**Total**: ~500 titles spanning 1930s-2024

---

## 🛠️ Technical Details

**Vector Search**:
- Embeddings: 384-dimensional vectors
- Index: Pinecone (serverless)
- Similarity: Cosine distance

**Query Strategy**:
- 3x title emphasis during embedding
- Minimum similarity threshold: 0.25
- Top-K retrieval: 5 results

**LLM**:
- Model: Zephyr-7B-Beta
- Max tokens: 1024
- Temperature: 0.7 (default)

---

## 📜 License

MIT License - feel free to use for your own projects!

---

## 🤝 Contributing

Issues and pull requests welcome! This is a portfolio/demo project.

---

## 📧 Contact

For questions about deployment or the project, open an issue on GitHub.

---

**Built with ❤️ using Streamlit, Pinecone, and HuggingFace**
