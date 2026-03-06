import os, streamlit as st, PyPDF2, io, requests, re, graphviz, faiss, numpy as np
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer

# --- 1. THE ONLINE BRAIN (Using Gemini for Cloud) ---
class OnlineChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"

    def get_response(self, prompt):
        # IF-ELSE Logic for Flowcharts
        if any(w in prompt.lower() for w in ["flowchart", "diagram", "process"]):
            prompt += "\nINSTRUCTION: Return a Graphviz DOT block 'digraph G { ... }'."
        
        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(self.url, json=payload, timeout=30)
            return r.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"Cloud Brain Error: {e}"

# --- 2. THE POWERPOINT ENGINE ---
def create_ppt(text):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "MAPS Academy Module"
    slide.placeholders[1].text = text[:1000]
    ppt_io = io.BytesIO()
    prs.save(ppt_io)
    return ppt_io.getvalue()

# --- 3. UI SETUP ---
st.set_page_config(page_title="MAPS Cloud AI", layout="wide")
sree_icon, enquirer_icon = "👩‍🏫", "👨‍🎓"

if 'messages' not in st.session_state: st.session_state.messages = []
if 'index' not in st.session_state:
    st.session_state.index, st.session_state.embedder = None, SentenceTransformer('all-MiniLM-L6-v2')

# --- 4. SIDEBAR (Cloud Settings) ---
with st.sidebar:
    st.title("👩‍💻 Cloud Admin")
    # In Cloud, we need an API Key. You can get one for free at Google AI Studio.
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    st.divider()
    files = st.file_uploader("Upload Training Manuals", accept_multiple_files=True)
    if st.button("🚀 Process Manuals") and files:
        chunks, sources = [], []
        for f in files:
            if f.name.endswith('.pdf'):
                pdf = PyPDF2.PdfReader(f)
                for i, p in enumerate(pdf.pages):
                    txt = p.extract_text()
                    if txt: 
                        chunks.append(txt)
                        sources.append(f"{f.name} (p.{i+1})")
        
        if chunks:
            embs = st.session_state.embedder.encode(chunks)
            st.session_state.index = faiss.IndexFlatL2(embs.shape[1])
            st.session_state.index.add(np.array(embs).astype('float32'))
            st.session_state.docs, st.session_state.sources = chunks, sources
            st.success("Manuals Processed!")

    if st.button("📝 Graduation Quiz"):
        if api_key:
            st.session_state.chatbot = OnlineChatbot(api_key)
            st.session_state.current_quiz = st.session_state.chatbot.get_response("Generate a 3-question MCQ based on the manuals.")

# --- 5. CHAT INTERFACE ---
st.title("📂 MAPS Academy: Online Assistant")

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=(sree_icon if m["role"]=="assistant" else enquirer_icon)):
        st.markdown(m["content"])

selected_lang = st.segmented_control("Language:", options=["English", "Hindi", "Telugu"], default="English")

if prompt := st.chat_input("Ask Sree..."):
    if not api_key:
        st.warning("Please enter your API Key in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=enquirer_icon): st.markdown(prompt)

        # RAG Search
        context = ""
        if st.session_state.index:
            q_emb = st.session_state.embedder.encode([prompt])
            D, I = st.session_state.index.search(np.array(q_emb).astype('float32'), k=3)
            context = "\n".join([st.session_state.docs[idx] for idx in I[0]])

        # Assistant Response
        with st.chat_message("assistant", avatar=sree_icon):
            st.markdown(":blue[**Sree**]")
            chatbot = OnlineChatbot(api_key)
            translated_prompt = f"Context: {context}\n\nQuestion: {prompt}. Answer only in {selected_lang}."
            response = chatbot.get_response(translated_prompt)
            
            clean = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
            clean = re.sub(r'(?i)digraph.*?\{.*?\}', '', clean, flags=re.DOTALL).strip()
            st.markdown(clean)
            
            if "digraph" in response.lower():
                match = re.search(r'digraph.*?\{.*?\}', response, re.DOTALL | re.IGNORECASE)
                if match: st.graphviz_chart(match.group(0))

            st.session_state.messages.append({"role": "assistant", "content": clean})

if "current_quiz" in st.session_state:
    st.divider()
    st.subheader("🎓 Graduation Quiz")
    st.write(st.session_state.current_quiz)
    if st.button("Close Quiz"): del st.session_state.current_quiz; st.rerun()
