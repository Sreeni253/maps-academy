import os, streamlit as st, PyPDF2, io, requests, re, base64, faiss, numpy as np
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from gtts import gTTS

# --- 1. THE "SOUL" (Persistent RAG) ---
st.set_page_config(page_title="MAPS Master", layout="wide")

# We use st.cache_resource so the "Brain" only loads ONCE and never resets
@st.cache_resource
def load_brain():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_brain()

# Initialize Session Memory
if 'msg' not in st.session_state: st.session_state.msg = []
if 'docs' not in st.session_state: st.session_state.docs = []
if 'idx' not in st.session_state: st.session_state.idx = None

# --- 2. THE HANDSHAKE TOOLS ---
def get_sree_response(api, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api}"
    try:
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30).json()
        return r['candidates'][0]['content']['parts'][0]['text']
    except: return "⚠️ Sree's connection timed out. Please try again."

def speak(text, lang):
    l_code = {"English": "en", "Hindi": "hi", "Telugu": "te"}.get(lang, "en")
    try:
        fp = io.BytesIO(); gTTS(text=text[:500], lang=l_code).write_to_fp(fp); fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)
    except: pass

# --- 3. SIDEBAR (The Diagnostic Handshake) ---
with st.sidebar:
    st.title("👩‍🏫 MAPS Admin")
    api = st.text_input("Gemini API Key:", type="password")
    
    st.divider()
    uploaded_files = st.file_uploader("Upload Training PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 FORCE PROCESS"):
        if uploaded_files:
            all_text = []
            for f in uploaded_files:
                # Direct Handshake: No temporary saving, read immediately
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t: all_text.append(t)
            
            if all_text:
                st.session_state.docs = all_text
                # Create the Vector Memory
                embeddings = model.encode(all_text)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings).astype('float32'))
                st.session_state.idx = index
                st.success(f"🤝 Handshake Success! Sree read {len(all_text)} pages.")
                # Show a snippet to PROVE it was read
                st.caption(f"Snippet: {all_text[0][:100]}...")
            else:
                st.error("❌ Handshake Failed: PDF is unreadable or empty.")

# --- 4. CHAT INTERFACE ---
st.title("📂 MAPS Academy Assistant")

# Diagnostic visual: Is Sree's memory full or empty?
if st.session_state.idx:
    st.sidebar.success("✅ Brain: Loaded")
else:
    st.sidebar.error("❌ Brain: Empty (Upload Manuals)")

for m in st.session_state.msg:
    with st.chat_message(m["role"]): st.markdown(m["content"])

lang = st.segmented_control("Language:", ["English", "Hindi", "Telugu"], default="English")

if prompt := st.chat_input("Ask Sree..."):
    if not api:
        st.error("Please enter API Key.")
    else:
        st.session_state.msg.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Context Handshake
        context = ""
        if st.session_state.idx:
            # Search the memory
            query_emb = model.encode([prompt])
            D, I = st.session_state.idx.search(np.array(query_emb).astype('float32'), k=2)
            context = "\n".join([st.session_state.docs[i] for i in I[0]])

        with st.chat_message("assistant"):
            full_prompt = f"Context: {context}\n\nUser: {prompt}\n(Answer in {lang} as Sree from MAPS Academy)"
            response = get_sree_response(api, full_prompt)
            st.markdown(response)
            speak(response, lang)
            st.session_state.msg.append({"role": "assistant", "content": response})
