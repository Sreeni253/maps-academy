import os, streamlit as st, PyPDF2, io, requests, re, base64, faiss, numpy as np
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from gtts import gTTS

# --- 1. CONFIG & BRAIN ---
st.set_page_config(page_title="MAPS Master", layout="wide")
if 'msg' not in st.session_state: st.session_state.msg = []
if 'idx' not in st.session_state:
    st.session_state.idx, st.session_state.model = None, SentenceTransformer('all-MiniLM-L6-v2')
if 'docs' not in st.session_state: st.session_state.docs = []

def get_sree_response(api, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api}"
    try:
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30).json()
        if 'candidates' in r: return r['candidates'][0]['content']['parts'][0]['text']
        return "⚠️ Brain Error: No response. Check API Key or Safety Filters."
    except Exception as e: return f"Error: {e}"

# --- 2. TOOLS (Voice & PPT) ---
def speak(text, lang):
    l_code = {"English": "en", "Hindi": "hi", "Telugu": "te"}.get(lang, "en")
    try:
        fp = io.BytesIO(); gTTS(text=text[:500], lang=l_code).write_to_fp(fp); fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)
    except: pass

# --- 3. UI SIDEBAR (With Deep Processing) ---
with st.sidebar:
    st.title("👩‍💻 MAPS Admin")
    api = st.text_input("Gemini API Key:", type="password")
    
    st.divider()
    files = st.file_uploader("Upload Training Manuals", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 Deep Process Manuals") and files:
        with st.spinner("Sree is reading every page..."):
            all_chunks = []
            for f in files:
                try:
                    # Reading the file immediately into bytes to prevent cloud timeout
                    bytes_data = f.read()
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 10:
                            all_chunks.append(f"Source: {f.name} (p.{i+1})\n\n{text}")
                except Exception as e: st.error(f"Error reading {f.name}: {e}")
            
            if all_chunks:
                # Encoding into FAISS Memory
                embs = st.session_state.model.encode(all_chunks)
                st.session_state.idx = faiss.IndexFlatL2(embs.shape[1])
                st.session_state.idx.add(np.array(embs).astype('float32'))
                st.session_state.docs = all_chunks
                st.success(f"✅ Sree memorized {len(all_chunks)} pages!")
            else:
                st.warning("⚠️ No text found. Are the PDFs scanned images? (Sree needs digital text)")

# --- 4. CHAT ---
st.title("📂 MAPS Academy Assistant")
for m in st.session_state.msg:
    with st.chat_message(m["role"]): st.markdown(m["content"])

lang = st.segmented_control("Language:", ["English", "Hindi", "Telugu"], default="English")

if prompt := st.chat_input("Ask Sree..."):
    if not api: st.warning("API Key missing")
    else:
        st.session_state.msg.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        ctx = ""
        if st.session_state.idx:
            # Search context
            D, I = st.session_state.idx.search(np.array(st.session_state.model.encode([prompt])).astype('float32'), k=3)
            ctx = "\n---\n".join([st.session_state.docs[i] for i in I[0]])
            
        with st.chat_message("assistant"):
            full_p = f"You are Sree, instructor at MAPS Academy. Use this context: {ctx}\n\nQuestion: {prompt}. Answer in {lang}."
            res = get_sree_response(api, full_p)
            st.markdown(res); speak(res, lang)
            st.session_state.msg.append({"role": "assistant", "content": res})
