import os, streamlit as st, PyPDF2, io, requests, subprocess, shutil, re, graphviz, faiss, numpy as np, platform
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer

# --- 1. SYSTEM SETUP & PATHS ---
SYSTEM = platform.system()
if SYSTEM == "Windows":
    LIBRARY_PATH = r"C:\MAPS_Library"
    import pyttsx3
    voice_engine = pyttsx3.init()
    os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
else:
    LIBRARY_PATH = os.path.expanduser("~/Desktop/MAPS_Library")

if not os.path.exists(LIBRARY_PATH): os.makedirs(LIBRARY_PATH)

# --- 2. THE BRAIN (Universal Logic) ---
class UniversalChatbot:
    def __init__(self, selected_model):
        self.model_name = selected_model
        self.url = "http://localhost:11434/api/generate"
        self.documents = []

    def get_response(self, prompt):
        if any(w in prompt.lower() for w in ["flowchart", "diagram", "process"]):
            prompt += "\nINSTRUCTION: Return a Graphviz DOT block 'digraph G { ... }'."
        try:
            r = requests.post(self.url, json={"model": self.model_name, "prompt": prompt, "stream": False}, timeout=90)
            return r.json().get('response', "Model error.")
        except: return "Ollama is not running."

# --- 3. THE ENGINES (Voice, PPT, Infographic) ---
def universal_speak(text, lang="English"):
    clean_text = text.replace('**', '').replace('#', '')
    if SYSTEM == "Windows":
        voice_engine.say(clean_text); voice_engine.runAndWait()
    else:
        voice = {"Hindi": "Lekha", "Telugu": "Vani"}.get(lang, "Rishi")
        subprocess.Popen(["say", "-v", voice, "-r", "160", clean_text])

def create_ppt(text):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "MAPS Training Module"
    slide.placeholders[1].text = text[:1000] # Simplified for brevity
    ppt_io = io.BytesIO()
    prs.save(ppt_io)
    return ppt_io.getvalue()

# --- 4. DATA PROCESSING ---
def extract_text(file_path):
    name, data = os.path.basename(file_path), []
    try:
        if name.endswith('.pdf'):
            pdf = PyPDF2.PdfReader(file_path)
            for i, p in enumerate(pdf.pages):
                if p.extract_text(): data.append({"text": p.extract_text(), "source": f"{name} (p.{i+1})"})
        elif name.endswith('.docx'):
            data.append({"text": "\n".join([p.text for p in Document(file_path).paragraphs]), "source": name})
    except: pass
    return data

# --- 5. UI & SIDEBAR ---
st.set_page_config(page_title="MAPS Academy Master", layout="wide")
sree_icon, enquirer_icon = "👩‍🏫", "👨‍🎓"

if 'messages' not in st.session_state: st.session_state.messages = []
if 'index' not in st.session_state:
    st.session_state.index, st.session_state.embedder = None, SentenceTransformer('all-MiniLM-L6-v2')

with st.sidebar:
    st.title("👨‍🏫 MAPS Master Admin")
    sel_model = st.selectbox("Brain:", ["deepseek-r1:7b", "gemma2:9b"])
    if st.button("🛑 Stop Voice"):
        if SYSTEM == "Windows": voice_engine.stop()
        else: subprocess.run(["killall", "say"])
    
    st.divider()
    files = st.file_uploader("Upload Manuals", accept_multiple_files=True)
    if st.button("🚀 Sync & Process Library") and files:
        for f in files:
            with open(os.path.join(LIBRARY_PATH, f.name), "wb") as b: shutil.copyfileobj(f, b)
        st.session_state.index = None # Reset index to force reload
        st.success("Library Updated!")
        st.rerun()

    if st.button("📝 Generate Graduation Quiz"):
        if st.session_state.index:
            st.session_state.current_quiz = st.session_state.chatbot.get_response("Generate a 3-question MCQ quiz based on the manuals.")
        else: st.warning("No manuals processed.")

# Initialize Chatbot
if "chatbot" not in st.session_state: st.session_state.chatbot = UniversalChatbot(sel_model)

# --- 6. RAG INDEXING ---
if os.path.exists(LIBRARY_PATH) and st.session_state.index is None:
    lib_files = [f for f in os.listdir(LIBRARY_PATH) if not f.startswith('.')]
    if lib_files:
        chunks, sources = [], []
        for fn in lib_files:
            for it in extract_text(os.path.join(LIBRARY_PATH, fn)):
                chunks.append(it["text"]); sources.append(it["source"])
        if chunks:
            embs = st.session_state.embedder.encode(chunks)
            st.session_state.index = faiss.IndexFlatL2(embs.shape[1])
            st.session_state.index.add(np.array(embs).astype('float32'))
            st.session_state.docs, st.session_state.sources = chunks, sources

# --- 7. CHAT & LANGUAGE ---
st.title("📂 MAPS Academy Master Assistant")
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=(sree_icon if m["role"]=="assistant" else enquirer_icon)):
        st.markdown(m["content"])

selected_lang = st.segmented_control("Language:", options=["English", "Hindi", "Telugu"], default="English")

if prompt := st.chat_input("Ask Sree..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=enquirer_icon): st.markdown(prompt)

    context = ""
    if st.session_state.index:
        q_emb = st.session_state.embedder.encode([prompt])
        D, I = st.session_state.index.search(np.array(q_emb).astype('float32'), k=3)
        context = "\n".join([st.session_state.docs[idx] for idx in I[0]])

    with st.chat_message("assistant", avatar=sree_icon):
        translated_prompt = f"Context: {context}\n\nQuestion: {prompt}. Please respond only in {selected_lang}."
        resp = st.session_state.chatbot.get_response(translated_prompt)
        clean = re.sub(r'```.*?```', '', resp, flags=re.DOTALL)
        clean = re.sub(r'(?i)digraph.*?\{.*?\}', '', clean, flags=re.DOTALL).strip()
        
        st.markdown(f":blue[**Sree**]\n\n{clean}")
        universal_speak(clean, lang=selected_lang)
        
        if "digraph" in resp.lower():
            match = re.search(r'digraph.*?\{.*?\}', resp, re.DOTALL | re.IGNORECASE)
            if match: st.graphviz_chart(match.group(0))
            
        if "ppt" in prompt.lower():
            st.download_button("📥 Download PPT", create_ppt(clean), "lesson.pptx")

        st.session_state.messages.append({"role": "assistant", "content": clean})

if "current_quiz" in st.session_state:
    st.subheader("🎓 Graduation Quiz")
    st.write(st.session_state.current_quiz)
    if st.button("Close Quiz"): del st.session_state.current_quiz; st.rerun()
