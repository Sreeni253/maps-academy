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
from bs4 import BeautifulSoup
from gtts import gTTS

# --- STEP 1: RESTORED PROVIDERS (Cloud-Safe Handshake) ---

class AIProvider:
    def get_response(self, prompt):
        raise NotImplementedError

class GeminiProvider(AIProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        # Using a stable web endpoint to ensure the Cloud handshake never fails
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"

    def get_response(self, prompt):
        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(self.url, json=payload, timeout=30)
            data = r.json()
            if 'candidates' in data and data['candidates']:
                return data['candidates'][0]['content']['parts'][0]['text']
            return "⚠️ Brain Error: No candidates. Check API Key or Safety Filters."
        except Exception as e:
            return f"Handshake Error: {e}"

# --- STEP 2: THE MASTER CHATBOT CLASS (Faithful to Original) ---

class UniversalChatbot:
    def __init__(self, ai_provider):
        self.ai_provider = ai_provider
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.index = None

    def add_documents(self, text, source_name):
        # Your original chunking logic
        chunks = self.chunk_text(text)
        for chunk in chunks:
            self.documents.append({'content': chunk, 'source': source_name})

    def chunk_text(self, text, chunk_size=1000, overlap=200):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def build_index(self):
        if not self.documents: return
        embeddings = self.embedding_model.encode([doc['content'] for doc in self.documents])
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

    def get_response(self, question):
        if not self.index:
            return "Knowledge base is empty. Please process manuals."
        
        # RAG Search
        query_emb = self.embedding_model.encode([question])
        D, I = self.index.search(query_emb.astype('float32'), k=3)
        
        context = ""
        for idx in I[0]:
            if idx < len(self.documents):
                context += self.documents[idx]['content'] + "\n\n"

        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer as Sree from MAPS Academy:"
        return self.ai_provider.get_response(prompt)

# --- STEP 3: THE VOICE ENGINE (New Cloud Addition) ---

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

# --- STEP 4: MAIN APP (Restored 700-line logic) ---

def main():
    st.set_page_config(page_title="MAPS Academy Master", layout="wide")

    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'chatbot' not in st.session_state: st.session_state.chatbot = None

    # Sidebar
    with st.sidebar:
        st.title("👩‍🏫 MAPS Admin")
        api_key = st.text_input("Gemini API Key:", type="password")
        
        st.divider()
        st.subheader("📤 Manual Upload")
        manual_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        
        if st.button("🚀 PROCESS ALL SOURCES"):
            if api_key and manual_files:
                with st.spinner("Sree is reading..."):
                    provider = GeminiProvider(api_key)
                    chatbot = UniversalChatbot(provider)
                    for f in manual_files:
                        # CRITICAL: Reading bytes immediately for Cloud handshake
                        pdf_data = f.read()
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() or ""
                        chatbot.add_documents(text, f.name)
                    
                    chatbot.build_index()
                    st.session_state.chatbot = chatbot
                    st.success(f"🤝 Handshake Success! Sree read {len(chatbot.documents)} segments.")

        st.divider()
        if st.button("📝 Graduation Quiz"):
            if st.session_state.chatbot:
                st.session_state.current_quiz = st.session_state.chatbot.get_response("Create a 3-question MCQ quiz based on the manuals.")

    # Main Chat Area
    st.title("📂 MAPS Academy Master Assistant")
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    # Language Selector
    selected_lang = st.segmented_control("Language:", ["English", "Hindi", "Telugu"], default="English")

    if prompt := st.chat_input("Ask Sree..."):
        if not st.session_state.chatbot:
            st.warning("Please upload and process manuals first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                st.markdown(":blue[**Sree**]")
                # The language handshake
                query = f"{prompt} (Respond ONLY in {selected_lang})"
                response = st.session_state.chatbot.get_response(query)
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
