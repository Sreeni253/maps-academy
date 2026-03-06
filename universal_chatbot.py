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
import pandas as pd
from pptx import Presentation
from google.oauth2 import service_account
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import time
from gtts import gTTS

# --- 1. THE RESTORED BRAIN (Fixing the 'candidates' error) ---

class GeminiProvider:
    def __init__(self, api_key):
        # We use the requests method to ensure a stable handshake on Streamlit Cloud
        self.api_key = api_key
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    def get_response(self, prompt):
        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(self.url, json=payload, timeout=30)
            data = r.json()
            # Safety check: Fixes the 'candidates' crash
            if 'candidates' in data and len(data['candidates']) > 0:
                return data['candidates'][0]['content']['parts'][0]['text']
            elif 'error' in data:
                return f"⚠️ API Error: {data['error']['message']}"
            return "⚠️ Sree's brain returned an empty response. Check your API Key."
        except Exception as e:
            return f"Cloud Brain Error: {e}"

# --- 2. THE RESTORED CHATBOT ENGINE (Full RAG & Drive Logic) ---

class UniversalChatbot:
    def __init__(self, ai_provider, credentials_json=None):
        self.ai_provider = ai_provider
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
        self.service = None
        if credentials_json:
            self._setup_service_account(credentials_json)
        self.documents = []
        self.index = None

    def _setup_service_account(self, credentials_json):
        try:
            credentials = service_account.Credentials.from_service_account_info(
                credentials_json, scopes=self.SCOPES
            )
            self.service = build('drive', 'v3', credentials=credentials)
        except Exception as e:
            st.error(f"Drive Setup Error: {e}")

    def scrape_website(self, url):
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.content, 'html.parser')
            return soup.get_text()[:5000]
        except: return ""

    def chunk_text(self, text, chunk_size=800, overlap=150):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def get_response(self, question):
        """Standard get_response for RAG"""
        if not self.index:
            return "Please load your training modules first."
        
        # Search Memory
        query_emb = self.embedding_model.encode([question])
        D, I = self.index.search(query_emb.astype('float32'), k=3)
        
        context = ""
        for idx in I[0]:
            if idx < len(self.documents):
                context += self.documents[idx]['content'] + "\n\n"

        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer naturally as Sree:"
        return self.ai_provider.get_response(prompt)

    # ALIGNMENT FIX: Maps 'ask_question' to 'get_response' to prevent crashes
    def ask_question(self, question):
        return self.get_response(question)

# --- 3. THE NEW CLOUD MOUTH (Browser Voice) ---

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

# --- 4. THE RESTORED UI (Faithful to Original) ---

def main():
    st.set_page_config(page_title="Universal AI Chatbot", layout="wide")

    # Security Gate
    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🤖 Universal AI Chatbot")
        code = st.text_input("Enter Access Code:", type="password")
        if st.button("Unlock"):
            if code == "Sree2026":
                st.session_state.authenticated = True
                st.rerun()
            else: st.error("Incorrect code.")
        return

    # Persistent State
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'chatbot' not in st.session_state: st.session_state.chatbot = None

    # THE ORIGINAL SIDEBAR (Drive, Web, Local)
    with st.sidebar:
        st.header("🧠 Cloud Brain")
        api_key = st.text_input("Gemini API Key:", type="password")
        
        st.divider()
        st.header("🎓 Academy Controls")
        if st.button("📊 Step 1: Presentation"): st.session_state.academy_step = "Step 1"
        if st.button("👨‍🏫 Step 2: Tutor Mode"): st.session_state.academy_step = "Step 2"
        if st.button("🎓 Step 3: Graduation Quiz"):
            if st.session_state.chatbot:
                st.session_state.current_quiz = st.session_state.chatbot.ask_question("Generate an MCQ quiz.")

        st.divider()
        st.header("📤 Manual Upload")
        manual_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        
        if st.button("🚀 Process All Sources"):
            if api_key and manual_files:
                with st.spinner("Sree is reading..."):
                    provider = GeminiProvider(api_key)
                    chatbot = UniversalChatbot(provider)
                    all_chunks = []
                    for f in manual_files:
                        pdf_data = f.read()
                        reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
                        text = "".join([p.extract_text() for p in reader.pages])
                        chunks = chatbot.chunk_text(text)
                        for c in chunks:
                            chatbot.documents.append({'content': c, 'source': f.name})
                            all_chunks.append(c)
                    
                    if all_chunks:
                        embs = chatbot.embedding_model.encode(all_chunks)
                        chatbot.index = faiss.IndexFlatL2(embs.shape[1])
                        chatbot.index.add(np.array(embs).astype('float32'))
                        st.session_state.chatbot = chatbot
                        st.success(f"🤝 Handshake Success! {len(all_chunks)} chunks loaded.")

    # --- BRANDED CHAT INTERFACE ---
    st.title("📂 MAPS Academy Master Assistant")
    sree_icon = "👩‍🏫"
    enquirer_icon = "💡"

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=(sree_icon if m["role"]=="assistant" else enquirer_icon)):
            st.markdown(m["content"])

    # Language Switch (Visual Placement Fix)
    selected_lang = st.segmented_control("Language:", ["English", "Hindi", "Telugu"], default="English")

    if prompt := st.chat_input("Ask Sree..."):
        if not st.session_state.chatbot:
            st.warning("Please upload and process manuals first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=enquirer_icon): st.markdown(prompt)

            with st.chat_message("assistant", avatar=sree_icon):
                st.markdown(":blue[**Sree**]")
                # The language instruction is secret
                translated_prompt = f"{prompt} (Respond ONLY in {selected_lang})"
                response = st.session_state.chatbot.ask_question(translated_prompt)
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
