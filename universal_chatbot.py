import os, streamlit as st, PyPDF2, io, requests, re, graphviz, faiss, numpy as np, base64
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from gtts import gTTS

# --- 1. CLOUD BRAIN (Gemini Adapter) ---
class CloudChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    def get_response(self, prompt):
        if any(w in prompt.lower() for w in ["flowchart", "diagram", "steps"]):
            prompt += "\nINSTRUCTION: Return a Graphviz DOT block starting with 'digraph G {'."
        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(self.url, json=payload, timeout=30)
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e: return f"Cloud Brain Error: {e}"

# --- 2. VOICE & PPT ENGINES ---
def universal_speak(text, lang="English"):
    lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
    try:
        # Generate audio in memory using Google TTS
        tts = gTTS(text=text[:500], lang=lang_map.get(lang, "en")) 
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        # Autoplay audio in the browser
        b64 = base64.b64encode(fp.read()).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_allow_html=True)
    except: pass

def create_ppt(text):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "MAPS Academy Training"
    slide.placeholders[1].text = text[:1000]
    ppt_io = io.BytesIO()
    prs.save(ppt_io)
    return ppt_io.getvalue()

# --- 3. UI SETUP ---
st.set_page_config(page_title="MAPS Academy Cloud", layout="wide")
sree_icon, enquirer_icon = "👩‍🏫", "👨‍🎓"

if 'messages' not in st.session_state: st.session_state.messages = []
if 'index' not in st.session_state:
    st.session_state.index, st.session_state.embedder = None, SentenceTransformer('all-MiniLM-L6-v2')

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("👩‍💻 Cloud Admin")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    files = st.file_uploader("Upload Manuals", accept_multiple_files=True)
    if st.button("🚀 Process Manuals") and files:
        chunks, src = [], []
        for f in files:
            if f.name.endswith('.pdf'):
                pdf = PyPDF2.PdfReader(f)
                for i, p in enumerate(pdf.pages):
                    txt = p.extract_text()
                    if txt: chunks.append(txt); src.append(f"{f.name} (p.{i+1})")
        if chunks:
            embs = st.session_state.embedder.encode(chunks)
            st.session_state.index = faiss.IndexFlatL2(embs.shape[1])
            st.session_state.index.add(np.array(embs).astype('float32'))
            st.session_state.docs, st.session_state.sources = chunks, src
            st.success("Manuals Processed!")
    
    if st.button("📝 Graduation Quiz") and api_key:
        bot = CloudChatbot(api_key)
        st.session_state.current_quiz = bot.get_response("Generate a 3-question MCQ quiz based on the uploaded data.")

# --- 5. MAIN CHAT ---
st.title("📂 MAPS Academy Master Assistant")
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=(sree_icon if m["role"]=="assistant" else enquirer_icon)):
        st.markdown(m["content"])

selected_lang = st.segmented_control("Language:", options=["English", "Hindi", "Telugu"], default="English")

if prompt := st.chat_input("Ask Sree..."):
    if not api_key: st.warning("Enter API Key!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=enquirer_icon): st.markdown(prompt)

        context = ""
        if st.session_state.index:
            q_emb = st.session_state.embedder.encode([prompt])
            D, I = st.session_state.index.search(np.array(q_emb).astype('float32'), k=2)
            context = "\n".join([st.session_state.docs[idx] for idx in I[0]])

        with st.chat_message("assistant", avatar=sree_icon):
            st.markdown(":blue[**Sree**]")
            bot = CloudChatbot(api_key)
            resp = bot.get_response(f"Context: {context}\n\nQuestion: {prompt}. Answer only in {selected_lang}.")
            clean = re.sub(r'```.*?```', '', resp, flags=re.DOTALL)
            clean = re.sub(r'(?i)digraph.*?\{.*?\}', '', clean, flags=re.DOTALL).strip()
            st.markdown(clean)
            universal_speak(clean, lang=selected_lang) # Auto-voice
            
            if "digraph" in resp.lower():
                match = re.search(r'digraph.*?\{.*?\}', resp, re.DOTALL | re.IGNORECASE)
                if match: st.graphviz_chart(match.group(0))
            if "ppt" in prompt.lower():
                st.download_button("📥 Download PPT", create_ppt(clean), "lesson.pptx")
            st.session_state.messages.append({"role": "assistant", "content": clean})

if "current_quiz" in st.session_state:
    st.divider(); st.subheader("🎓 Graduation Quiz"); st.write(st.session_state.current_quiz)
    if st.button("Close Quiz"): del st.session_state.current_quiz; st.rerun()
