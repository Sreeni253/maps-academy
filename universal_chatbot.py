import sys
import os
import streamlit as st
import PyPDF2
import io
import json
import re
import base64
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from docx import Document
from pptx import Presentation
from gtts import gTTS
from bs4 import BeautifulSoup

# --- 1. THE ORIGINAL BRAIN ADAPTERS ---
class UniversalChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        # Restore the specific Gemini 1.5 Flash endpoint used in your original
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        self.documents = []

    def get_response(self, prompt):
        # Restore the Flowchart instructions
        if any(w in prompt.lower() for w in ["flowchart", "diagram", "process"]):
            prompt += "\nINSTRUCTION: Always include a Graphviz DOT block starting with 'digraph G {'."
        
        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(self.url, json=payload, timeout=30)
            data = r.json()
            if 'candidates' in data and data['candidates']:
                return data['candidates'][0]['content']['parts'][0]['text']
            return "⚠️ Sree's brain is offline. Check API Key or Safety Filters."
        except Exception as e:
            return f"Connection Error: {e}"

# --- 2. THE MULTILINGUAL MOUTH ---
def universal_speak(text, lang="English"):
    """Cloud-safe voice handshake for Lenovo browser."""
    lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
    try:
        clean_text = re.sub(r'[*#_]', '', text[:500]) 
        tts = gTTS(text=clean_text, lang=lang_map.get(lang, "en"))
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_allow_html=True)
    except:
        pass

# --- 3. SYSTEM INITIALIZATION ---
st.set_page_config(page_title="MAPS Academy Master", layout="wide")
sree_icon, enquirer_icon = "👩‍🏫", "👨‍🎓"

# Persistent Global State
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'index' not in st.session_state:
    with st.spinner("⏳ Sree is initializing her memory..."):
        st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        st.session_state.index = None
if 'docs' not in st.session_state:
    st.session_state.docs = []

# --- 4. THE ORIGINAL FULL SIDEBAR ---
with st.sidebar:
    st.title("👩‍🏫 MAPS Admin Panel")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    st.divider()
    st.subheader("🌐 Knowledge Sources")
    web_urls = st.text_area("Website URLs (Optional):")
    
    st.divider()
    st.subheader("📤 Local Training Manuals")
    uploaded_files = st.file_uploader("Upload PDF Manuals", accept_multiple_files=True, type=['pdf'])
    
    # THE CRITICAL HANDSHAKE BUTTON
    if st.button("🚀 PROCESS & INDEX ALL DATA"):
        if uploaded_files or web_urls:
            with st.spinner("Sree is reading and indexing your training modules..."):
                all_chunks = []
                
                # Reading PDF Bytes - Original Method
                for f in uploaded_files:
                    try:
                        pdf_data = f.read()
                        reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
                        for i, page in enumerate(reader.pages):
                            txt = page.extract_text()
                            if txt:
                                all_chunks.append(f"Source: {f.name} (p.{i+1})\n{txt}")
                    except Exception as e:
                        st.error(f"Error reading {f.name}: {e}")
                
                # Website Scraper - Original Method
                if web_urls:
                    for url in web_urls.split('\n'):
                        url = url.strip()
                        if url:
                            try:
                                res = requests.get(url, timeout=10)
                                soup = BeautifulSoup(res.text, 'html.parser')
                                all_chunks.append(f"Web Source: {url}\n{soup.get_text()[:2000]}")
                            except:
                                pass

                if all_chunks:
                    st.session_state.docs = all_chunks
                    # Vector Memory Build
                    embeddings = st.session_state.embedder.encode(all_chunks)
                    st.session_state.index = faiss.IndexFlatL2(embeddings.shape[1])
                    st.session_state.index.add(np.array(embeddings).astype('float32'))
                    st.success(f"🤝 Handshake Complete! Sree memorized {len(all_chunks)} segments.")
        else:
            st.warning("Please upload files or enter URLs first.")

    st.divider()
    st.subheader("🎓 Skill Validation")
    if st.button("📝 Generate Graduation Quiz"):
        if api_key and st.session_state.docs:
            with st.spinner("Preparing exam..."):
                bot = UniversalChatbot(api_key)
                quiz_query = "Based on the technical documents provided, generate a 3-question MCQ quiz with an Answer Key."
                st.session_state.quiz = bot.get_response(quiz_query)
        else:
            st.error("Index manuals and enter API Key first!")

# --- 5. MAIN INTERFACE ---
st.title("📂 MAPS Academy Master Assistant")

if not st.session_state.docs:
    st.warning("⚠️ Sree's knowledge base is empty. Upload and process manuals in the sidebar to begin.")

# Display History
for m in st.session_state.messages:
    avatar = sree_icon if m["role"] == "assistant" else enquirer_icon
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# Alignment: Language Selector
selected_lang = st.segmented_control(
    "Response Language:", 
    options=["English", "Hindi", "Telugu"], 
    default="English"
)

if prompt := st.chat_input("Ask Sree..."):
    if not api_key:
        st.error("API Key required!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=enquirer_icon):
            st.markdown(prompt)

        # THE DATA HANDSHAKE (RAG Retrieval)
        context = ""
        if st.session_state.index:
            q_emb = st.session_state.embedder.encode([prompt])
            D, I = st.session_state.index.search(np.array(q_emb).astype('float32'), k=3)
            context = "\n---\n".join([st.session_state.docs[idx] for idx in I[0]])

        # Assistant Logic
        with st.chat_message("assistant", avatar=sree_icon):
            st.markdown(":blue[**Sree**]")
            bot = UniversalChatbot(api_key)
            full_prompt = f"Context: {context}\n\nUser Question: {prompt}\n(Answer strictly in {selected_lang} as Sree from MAPS Academy)"
            response = bot.get_response(full_prompt)
            
            # Clean display
            clean_display = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
            st.markdown(clean_display)
            
            # Restore Flowchart Rendering
            if "digraph" in response.lower():
                match = re.search(r'digraph.*?\{.*?\}', response, re.DOTALL | re.IGNORECASE)
                if match:
                    st.graphviz_chart(match.group(0))
            
            # New Voice Integration
            universal_speak(clean_display, lang=selected_lang)
            
            st.session_state.messages.append({"role": "assistant", "content": clean_display})

# Show Graduation Quiz
if "quiz" in st.session_state:
    st.divider()
    st.subheader("🎓 Sree's Graduation Quiz")
    st.write(st.session_state.quiz)
    if st.button("Close Quiz"):
        del st.session_state.quiz
        st.rerun()
