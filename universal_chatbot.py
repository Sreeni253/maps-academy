import os, streamlit as st, PyPDF2, io, requests, re, base64, faiss, numpy as np, json
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from gtts import gTTS
from bs4 import BeautifulSoup

# --- 1. THE HEAVY-DUTY BRAIN ---
class UniversalChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    def get_response(self, prompt):
        # Restore Flowchart Instruction
        if any(w in prompt.lower() for w in ["flowchart", "diagram", "process"]):
            prompt += "\nINSTRUCTION: Always include a Graphviz DOT block starting with 'digraph G {'."
        try:
            r = requests.post(self.url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30).json()
            if 'candidates' in r:
                return r['candidates'][0]['content']['parts'][0]['text']
            return "⚠️ Sree's brain is offline. Check API Key or Safety Filters."
        except Exception as e: return f"Error: {e}"

# --- 2. MULTILINGUAL VOICE ENGINE ---
def universal_speak(text, lang="English"):
    """Plays audio through the Lenovo browser using Google TTS."""
    lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
    try:
        clean_text = re.sub(r'[*#_]', '', text[:500])
        tts = gTTS(text=clean_text, lang=lang_map.get(lang, "en"))
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}"></audio>'
        st.markdown(md, unsafe_allow_html=True)
    except: pass

# --- 3. UI INITIALIZATION ---
st.set_page_config(page_title="MAPS Academy Master", layout="wide")
sree_icon, enquirer_icon = "👩‍🏫", "👨‍🎓"

if 'messages' not in st.session_state: st.session_state.messages = []
if 'index' not in st.session_state:
    st.session_state.index, st.session_state.embedder = None, SentenceTransformer('all-MiniLM-L6-v2')
if 'docs' not in st.session_state: st.session_state.docs = []

# --- 4. THE MASTER SIDEBAR (700-Line Style) ---
with st.sidebar:
    st.title("👩‍🏫 MAPS Admin Panel")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    st.divider()
    st.subheader("🌐 Knowledge Sources")
    web_urls = st.text_area("Website URLs (One per line):")
    drive_id = st.text_input("Google Drive Folder ID:")
    
    st.divider()
    st.subheader("📤 Local Training Manuals")
    files = st.file_uploader("Upload PDF Manuals", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 PROCESS & INDEX ALL DATA"):
        if files or web_urls:
            with st.spinner("Sree is memorizing the modules..."):
                all_chunks = []
                # 1. Process PDFs
                for f in files:
                    pdf_data = f.read()
                    reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
                    for i, page in enumerate(reader.pages):
                        txt = page.extract_text()
                        if txt: all_chunks.append(f"Manual: {f.name} (p.{i+1})\n{txt}")
                
                # 2. Process Websites (Simple Scraper)
                if web_urls:
                    for url in web_urls.split('\n'):
                        try:
                            res = requests.get(url.strip(), timeout=10)
                            soup = BeautifulSoup(res.text, 'html.parser')
                            all_chunks.append(f"Web: {url}\n{soup.get_text()[:2000]}")
                        except: pass

                if all_chunks:
                    st.session_state.docs = all_chunks
                    embeddings = st.session_state.embedder.encode(all_chunks)
                    st.session_state.index = faiss.IndexFlatL2(embeddings.shape[1])
                    st.session_state.index.add(np.array(embeddings).astype('float32'))
                    st.success(f"🤝 Handshake Complete! {len(all_chunks)} segments indexed.")
        else:
            st.warning("Please upload files or enter URLs first.")

    st.divider()
    st.subheader("🎓 Skill Validation")
    if st.button("📝 Generate Graduation Quiz"):
        if api_key and st.session_state.docs:
            bot = UniversalChatbot(api_key)
            st.session_state.quiz = bot.get_response("Based on the manuals, create a 3-question MCQ quiz with an Answer Key.")
        else: st.error("Need API Key and Processed Manuals!")

# --- 5. CHAT INTERFACE ---
st.title("📂 MAPS Academy Master Assistant")

if not st.session_state.docs:
    st.warning("⚠️ Sree's memory is empty. Use the sidebar to upload training manuals.")

for m in st.session_state.messages:
    avatar = sree_icon if m["role"] == "assistant" else enquirer_icon
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# Alignment: Language Switch right above chat bar
selected_lang = st.segmented_control("Response Language:", ["English", "Hindi", "Telugu"], default="English")

if prompt := st.chat_input("Ask Sree..."):
    if not api_key: st.error("API Key required!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=enquirer_icon): st.markdown(prompt)

        # Context Handshake
        context = ""
        if st.session_state.index:
            q_emb = st.session_state.embedder.encode([prompt])
            D, I = st.session_state.index.search(np.array(q_emb).astype('float32'), k=3)
            context = "\n---\n".join([st.session_state.docs[idx] for idx in I[0]])

        with st.chat_message("assistant", avatar=sree_icon):
            st.markdown(":blue[**Sree**]")
            bot = UniversalChatbot(api_key)
            full_prompt = f"Context: {context}\n\nUser: {prompt}. Answer strictly in {selected_lang}."
            response = bot.get_response(full_prompt)
            
            clean = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
            st.markdown(clean)
            universal_speak(clean, lang=selected_lang)
            
            if "digraph" in response.lower():
                match = re.search(r'digraph.*?\{.*?\}', response, re.DOTALL | re.IGNORECASE)
                if match: st.graphviz_chart(match.group(0))

            st.session_state.messages.append({"role": "assistant", "content": clean})

if "quiz" in st.session_state:
    st.divider()
    st.subheader("🎓 Sree's Graduation Quiz")
    st.write(st.session_state.quiz)
    if st.button("Close Quiz"): del st.session_state.quiz; st.rerun()
