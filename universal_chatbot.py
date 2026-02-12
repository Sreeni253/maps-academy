import streamlit as st
import PyPDF2
import io
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from docx import Document
import pandas as pd
from pptx import Presentation
from google.oauth2 import service_account
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import time
from streamlit_mic_recorder import mic_recorder

# AI Provider Adapters
class AIProvider:
    def get_response(self, prompt):
        raise NotImplementedError

class OpenAIProvider(AIProvider):
    def __init__(self, api_key):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
    
    def get_response(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content

class GeminiProvider(AIProvider):
    def __init__(self, api_key):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def get_response(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

class ClaudeProvider(AIProvider):
    def __init__(self, api_key):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def get_response(self, prompt):
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

# Universal Chatbot Logic
class UniversalChatbot:
    def __init__(self, ai_provider, credentials_json=None):
        self.ai_provider = ai_provider
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
        self.service = None
        if credentials_json:
            self._setup_service_account(credentials_json)
        self.documents = []
        self.embeddings = None
        self.index = None
    
    def _setup_service_account(self, credentials_json):
        try:
            credentials = service_account.Credentials.from_service_account_info(
                credentials_json, scopes=self.SCOPES
            )
            self.service = build('drive', 'v3', credentials=credentials)
        except Exception as e:
            st.error(f"Error setting up Google Drive: {e}")

    def scrape_website(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text()[:10000]
        except: return ""

    def list_files_in_folder(self, folder_id=None, folder_name=None):
        if not self.service: return []
        try:
            query = "mimeType='application/pdf' and trashed=false"
            if folder_id: query += f" and '{folder_id}' in parents"
            results = self.service.files().list(q=query, fields="files(id,name,mimeType)").execute()
            files = results.get('files', [])
            for f in files: f['file_type'] = 'PDF'
            return files
        except: return []

    def download_file_content(self, file_id, mime_type):
        try:
            return self.service.files().get_media(fileId=file_id).execute()
        except: return None

    def extract_text_from_file(self, file_content, file_name, file_type):
        if file_type == 'PDF': return self._extract_from_pdf(file_content)
        return ""

    def _extract_from_pdf(self, pdf_content):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        return "\n".join([page.extract_text() for page in pdf_reader.pages[:10]])

    def _extract_from_word(self, docx_content):
        doc = Document(io.BytesIO(docx_content))
        return "\n".join([p.text for p in doc.paragraphs[:50]])

   def load_technical_data(self, csv_content, file_name):
        """Processes CSVs as both searchable text AND technical dataframes"""
        try:
            df = pd.read_csv(io.BytesIO(csv_content))
            
            # 1. Store as searchable text for Sree's 'General Knowledge'
            text_data = f"Technical Data from {file_name}:\n" + df.to_string()
            chunks = self.chunk_text(text_data)
            for chunk in chunks:
                self.documents.append({
                    'source': file_name,
                    'file_type': 'Technical CSV',
                    'content': chunk
                })
            
            # 2. Store the actual table in memory for 'Calculations'
            if 'technical_tables' not in st.session_state:
                st.session_state.technical_tables = {}
            st.session_state.technical_tables[file_name] = df
            
            return True
        except Exception as e:
            st.error(f"Error loading technical model: {e}")
            return False
            
    def chunk_text(self, text, chunk_size=800, overlap=150):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks[:20]

    def process_all_sources(self, folder_id=None, folder_name=None, website_urls=None):
        self.documents = []
        all_processed = []
        # Logic for Drive and Web would go here as per your original code
        return all_processed

    def search_similar_chunks(self, query, k=3):
        if self.index is None: return []
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        return [{'content': self.documents[idx]['content'], 'source': self.documents[idx]['source'], 'file_type': self.documents[idx]['file_type']} for idx in indices[0] if idx < len(self.documents)]

    def get_response(self, question):
        if not self.index: return "Please load documents first."
        relevant_chunks = self.search_similar_chunks(question, k=3)
        context = "\n\n".join([c['content'] for c in relevant_chunks])
        sources = list(set([f"{c['source']} ({c['file_type']})" for c in relevant_chunks]))
        prompt = f"Answer using this context:\n{context}\n\nQuestion: {question}"
        answer = self.ai_provider.get_response(prompt)
        if sources: answer += f"\n\n**Sources:** {', '.join(sources)}"
        return answer

# --- MAIN APP INTERFACE ---
def main():
    # 1. Page Config
    st.set_page_config(page_title="Sree - MAPS Academy", page_icon="🌳")

    # 2. Security Gate
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🌳 Sree - MAPS Academy")
        st.markdown("### 🔐 Secure Student Access")
        password = st.text_input("Enter Academy Access Code:", type="password")
        if password == "Sree2026":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.info("Welcome. Please enter your student credentials to speak with Sree.")
            return

    # 3. Main Branded Header
    st.title("🌳 Sree - MAPS Academy")
    st.markdown("**Mapping and Advancing Professional Skills**")
    
    # 4. Sidebar Configuration
    with st.sidebar:
        st.header("🧠 Configuration")
        ai_choice = st.selectbox("Select AI Provider:", ["Gemini (Free)", "OpenAI (Paid)", "Claude (Paid)"])
        api_key = st.text_input("API Key:", type="password")
        
        st.header("📤 Training Modules")
        manual_files = st.file_uploader("Upload PDF or Word files:", accept_multiple_files=True)
        
        if st.button("🚀 Process All Sources"):
            if api_key:
                with st.spinner("Sree is gathering knowledge..."):
                    provider = GeminiProvider(api_key) if "Gemini" in ai_choice else OpenAIProvider(api_key)
                    chatbot = UniversalChatbot(provider)
                    all_processed = []
                    if manual_files:
                        for f in manual_files:
                            content = f.read()
                            text = chatbot._extract_from_pdf(content) if f.name.endswith('.pdf') else ""
                            if text:
                                chunks = chatbot.chunk_text(text)
                                for c in chunks:
                                    chatbot.documents.append({'source': f.name, 'file_type': 'PDF', 'content': c})
                                all_processed.append({'name': f.name, 'chunks': len(chunks)})
                    
                    if chatbot.documents:
                        all_chunks = [d['content'] for d in chatbot.documents]
                        chatbot.embeddings = chatbot.embedding_model.encode(all_chunks)
                        chatbot.index = faiss.IndexFlatL2(chatbot.embeddings.shape[1])
                        chatbot.index.add(chatbot.embeddings.astype('float32'))
                        st.session_state.chatbot = chatbot
                        st.session_state.processed_sources = all_processed
                        st.success("Ready!")
            else:
                st.error("API Key required.")

    # 5. Chat Interface
    sree_icon = "https://raw.githubusercontent.com/Sreeni253/maps-academy/main/kalpavruksha.png"
    enquirer_icon = "💡" 

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        avatar = sree_icon if message["role"] == "assistant" else enquirer_icon
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                st.markdown(":blue[**Sree**]") 
            st.markdown(message["content"])

    # 6. Input Section
    col1, col2 = st.columns([1, 9])
    with col1:
        audio = mic_recorder(start_prompt="🎤", stop_prompt="🛑", key='sree_mic')
    with col2:
        prompt = st.chat_input("Speak with Sree...")

    final_prompt = audio['text'] if audio and audio.get('text') else prompt

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user", avatar=enquirer_icon):
            st.markdown(final_prompt)
        
        if hasattr(st.session_state, 'chatbot'):
            with st.chat_message("assistant", avatar=sree_icon):
                st.markdown(":blue[**Sree**]")
                with st.spinner("Consulting modules..."):
                    response = st.session_state.chatbot.get_response(final_prompt)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
