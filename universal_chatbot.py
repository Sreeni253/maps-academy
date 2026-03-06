import os, streamlit as st, PyPDF2, io, requests, re, base64, faiss, numpy as np
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from gtts import gTTS

# --- 1. CONFIG & SYSTEM HANDSHAKE ---
st.set_page_config(page_title="MAPS Master", layout="wide")

# Persistent Memory Setup
if 'msg' not in st.session_state: st.session_state.msg = []
if 'docs' not in st.session_state: st.session_state.docs = []
if 'idx' not in st.session_state:
    with st.spinner("⏳ Sree is warming up her brain (Loading Embeddings)..."):
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
        st.session_state.idx = None

# --- 2. THE BRAIN (Gemini Cloud) ---
def get_sree_response(api, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api}"
    try:
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30).json()
        if 'candidates' in r: return r['candidates'][0]['content']['parts'][0]['text']
        return f"⚠️ Handshake Failed: Gemini returned an empty response. (Check API Key/Safety)"
    except Exception as e: return f"⚠️ Brain Connection Error: {e}"

# --- 3. THE VOICE (Cloud Autoplay) ---
def speak(text, lang):
    l_code = {"English": "en", "Hindi": "hi", "Telugu": "te"}.get(lang, "en")
    try:
        fp = io.BytesIO(); gTTS(text=text[:500], lang=l_code).write_to_fp(fp); fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}"></audio>'
        st.markdown(md, unsafe_allow_html=True)
    except: pass

# --- 4. THE SIDEBAR (Visualizing the Handshake) ---
with st.sidebar:
    st.title("👩‍🏫 MAPS Admin")
    api = st.text_input("Gemini API Key:", type="password")
    
    st.divider()
    st.subheader("📁 Manual Upload")
    files = st.file_uploader("Upload Training PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 Process & Index"):
        if not files:
            st.warning("❌ No files detected. Please select a PDF first.")
        else:
            all_chunks = []
            progress_bar = st.progress(0)
            for i, f in enumerate(files):
                try:
                    # Explicit Handshake: Reading raw bytes
                    reader = PyPDF2.PdfReader(io.BytesIO(f.read()))
                    file_text = ""
                    for p_num, page in enumerate(reader.pages):
                        txt = page.extract_text()
                        if txt:
                            chunk = f"FILE: {f.name} (Page {p_num+1})\n{txt}"
                            all_chunks.append(chunk)
                    progress_bar.progress((i + 1) / len(files))
                except Exception as e:
                    st.error(f"❌ Handshake Error with {f.name}: {e}")

            if all_chunks:
                # Store in Session State so it survives "Re-baking"
                st.session_state.docs = all_chunks
                embeddings = st.session_state.model.encode(all_chunks)
                st.session_state.idx = faiss.IndexFlatL2(embeddings.shape[1])
                st.session_state.idx.add(np.array(embeddings).astype('float32'))
                st.success(f"🤝 Handshake Successful! Sree memorized {len(all_chunks)} pages.")
            else:
                st.error("❌ Process Failed: No readable text found in those PDFs.")

# --- 5. CHAT INTERFACE ---
st.title("📂 MAPS Academy Assistant")

# Show manual status
if st.session_state.docs:
    st.info(f"📚 Current Knowledge Base: {len(st.session_state.docs)} document segments loaded.")
else:
    st.warning("⚠️ Sree is currently 'Blank'. Please upload manuals in the sidebar to begin.")

for m in st.session_state.msg:
    with st.chat_message(m["role"]): st.markdown(m["content"])

lang = st.segmented_control("Language:", ["English", "Hindi", "Telugu"], default="English")

if prompt := st.chat_input("Ask about your modules..."):
    if not api:
        st.error("🔑 Please enter your API Key in the sidebar.")
    else:
        st.session_state.msg.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Context Handshake (RAG)
        context = ""
        if st.session_state.idx:
            D, I = st.session_state.idx.search(np.array(st.session_state.model.encode([prompt])).astype('float32'), k=3)
            context = "\n---\n".join([st.session_state.docs[i] for i in I[0]])

        with st.chat_message("assistant"):
            full_prompt = f"Context: {context}\n\nUser Question: {prompt}\nAnswer in {lang}:"
            response = get_sree_response(api, full_prompt)
            st.markdown(response)
            speak(response, lang)
            st.session_state.msg.append({"role": "assistant", "content": response})
