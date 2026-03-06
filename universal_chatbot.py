import os, streamlit as st, PyPDF2, io, requests, re, graphviz, faiss, numpy as np, base64
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from gtts import gTTS

# --- 1. CLOUD BRAIN (Gemini Adapter with Safety Check) ---
class CloudChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    def get_response(self, prompt):
        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(self.url, json=payload, timeout=30)
            data = r.json()
            # FIX: Check if 'candidates' exists to prevent the crash you saw
            if 'candidates' in data and data['candidates']:
                return data['candidates'][0]['content']['parts'][0]['text']
            return "⚠️ Sree couldn't find an answer. Check your API key or manual content."
        except Exception as e: return f"Cloud Brain Error: {e}"

# --- 2. THE UI & SIDEBAR (Restoring Drive/Web) ---
st.set_page_config(page_title="MAPS Academy Cloud", layout="wide")

if 'messages' not in st.session_state: st.session_state.messages = []
if 'index' not in st.session_state:
    st.session_state.index, st.session_state.embedder = None, SentenceTransformer('all-MiniLM-L6-v2')

with st.sidebar:
    st.title("👩‍💻 Cloud Admin")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    # RESTORED: Google Drive & Web Fields
    st.header("🌐 External Sources")
    web_urls = st.text_area("Website URLs (one per line):")
    drive_id = st.text_input("Google Drive Folder ID (Optional):")
    
    st.header("📤 Local Training Modules")
    files = st.file_uploader("Upload Manuals", accept_multiple_files=True)
    
    if st.button("🚀 Process All Sources") and files:
        chunks, src = [], []
        for f in files:
            if f.name.endswith('.pdf'):
                pdf = PyPDF2.PdfReader(f)
                for i, p in enumerate(pdf.pages):
                    txt = p.extract_text()
                    if txt: chunks.append(txt); src.append(f"{f.name} (p.{i+1})")
        # Add indexing logic here...
        if chunks:
            embs = st.session_state.embedder.encode(chunks)
            st.session_state.index = faiss.IndexFlatL2(embs.shape[1])
            st.session_state.index.add(np.array(embs).astype('float32'))
            st.session_state.docs = chunks
            st.success(f"✅ Loaded {len(chunks)} chunks!")

# --- 3. MAIN CHAT & LANGUAGES ---
st.title("📂 MAPS Academy Master Assistant")
selected_lang = st.segmented_control("Language:", options=["English", "Hindi", "Telugu"], default="English")

if prompt := st.chat_input("Ask Sree..."):
    if not api_key: st.warning("Enter API Key!")
    else:
        # Retrieval Logic
        context = ""
        if st.session_state.index:
            q_emb = st.session_state.embedder.encode([prompt])
            D, I = st.session_state.index.search(np.array(q_emb).astype('float32'), k=2)
            context = "\n".join([st.session_state.docs[idx] for idx in I[0]])

        with st.chat_message("assistant"):
            bot = CloudChatbot(api_key)
            resp = bot.get_response(f"Context: {context}\n\nQuestion: {prompt}. Answer in {selected_lang}.")
            st.markdown(resp)
