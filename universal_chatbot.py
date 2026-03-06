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

# --- 1. AI PROVIDER ADAPTERS (Restored) ---

class GeminiProvider:
    def __init__(self, api_key):
        self.api_key = api_key
        # Stable endpoint for Cloud
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    def get_response(self, prompt):
        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(self.url, json=payload, timeout=30)
            data = r.json()
            # Safety Check for 'candidates'
            if 'candidates' in data and data['candidates']:
                return data['candidates'][0]['content']['parts'][0]['text']
            return "⚠️ Sree's brain is offline. Check API Key or Safety Filters."
        except Exception as e:
            return f"Cloud Brain Error: {e}"

# --- 2. THE MASTER CHATBOT ENGINE ---

class UniversalChatbot:
    def __init__(self, ai_provider):
        self.ai_provider = ai_provider
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.index = None

    def chunk_text(self, text, chunk_size=800, overlap=150):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip(): chunks.append(chunk)
            start = end - overlap
        return chunks

    def get_response(self, question):
        """The core RAG handshake logic"""
        if not self.index:
            return "Please load your training modules in the sidebar first."
        
        # Search Memory
        query_emb = self.embedding_model.encode([question])
        scores, indices = self.index.search(query_emb.astype('float32'), k=3)
        
        context = ""
        sources = []
        for idx in indices[0]:
            if idx < len(self.documents):
                context += self.documents[idx]['content'] + "\n\n"
                sources.append(self.documents[idx]['source'])

        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer naturally as Sree:"
        return self.ai_provider.get_response(prompt)

# --- 3. THE CLOUD MOUTH (Browser Audio) ---

def universal_speak(text, lang="English"):
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
    except: pass

# --- 4. MAIN APP LOGIC ---

def main():
    st.set_page_config(page_title="Universal AI Chatbot", page_icon="🤖", layout="wide")

    # Security Gate
    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🤖 MAPS Academy Access")
        code = st.text_input("Enter Access Code:", type="password")
        if st.button("Unlock"):
            if code == "Sree2026":
                st.session_state.authenticated = True
                st.rerun()
            else: st.error("Wrong code.")
        return

    # Initialize Session State
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'index' not in st.session_state: st.session_state.index = None

    # Sidebar
    with st.sidebar:
        st.header("🧠 Cloud Brain")
        api_key = st.text_input("Gemini API Key:", type="password")
        
        st.divider()
        st.header("🎓 Maps Academy Steps")
        if st.button("📊 Step 1: Presentation"): st.session_state.academy_step = "Step 1"
        if st.button("👨‍🏫 Step 2: Tutor Mode"): st.session_state.academy_step = "Step 2"
        if st.button("🎓 Step 3: Graduation Quiz"):
            if "chatbot" in st.session_state and st.session_state.chatbot.index:
                bot = st.session_state.chatbot
                st.session_state.current_quiz = bot.get_response("Generate a 3-question MCQ quiz based on the manuals.")
        
        st.divider()
        st.header("📤 Training Modules")
        manual_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        
        if st.button("🚀 Process All Sources"):
            if api_key and manual_files:
                with st.spinner("Sree is reading..."):
                    provider = GeminiProvider(api_key)
                    chatbot = UniversalChatbot(provider)
                    all_chunks = []
                    for f in manual_files:
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(f.read()))
                        for i, page in enumerate(pdf_reader.pages):
                            txt = page.extract_text()
                            if txt:
                                chunks = chatbot.chunk_text(txt)
                                for c in chunks:
                                    chatbot.documents.append({'source': f.name, 'content': c})
                                    all_chunks.append(c)
                    
                    if all_chunks:
                        embs = chatbot.embedding_model.encode(all_chunks)
                        chatbot.index = faiss.IndexFlatL2(embs.shape[1])
                        chatbot.index.add(np.array(embs).astype('float32'))
                        st.session_state.chatbot = chatbot
                        st.success(f"✅ Handshake Success! {len(all_chunks)} chunks loaded.")

    # Main Chat Area
    st.title("📂 MAPS Academy Master Assistant")
    
    sree_icon = "👩‍🏫"
    enquirer_icon = "💡"

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=(sree_icon if m["role"]=="assistant" else enquirer_icon)):
            st.markdown(m["content"])

    # Language Switcher
    selected_lang = st.segmented_control("Language:", ["English", "Hindi", "Telugu"], default="English")

    if prompt := st.chat_input("Ask Sree..."):
        if "chatbot" not in st.session_state:
            st.warning("Please process your manuals in the sidebar first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=enquirer_icon): st.markdown(prompt)

            with st.chat_message("assistant", avatar=sree_icon):
                st.markdown(":blue[**Sree**]")
                # Use translated prompt for the brain
                translated_prompt = f"{prompt} (Please respond ONLY in {selected_lang})"
                response = st.session_state.chatbot.get_response(translated_prompt)
                
                st.markdown(response)
                universal_speak(response, lang=selected_lang)
                st.session_state.messages.append({"role": "assistant", "content": response})

    if "current_quiz" in st.session_state:
        st.divider()
        st.subheader("🎓 Sree's Graduation Quiz")
        st.write(st.session_state.current_quiz)
        if st.button("Close Quiz"): del st.session_state.current_quiz; st.rerun()

if __name__ == "__main__":
    main()
