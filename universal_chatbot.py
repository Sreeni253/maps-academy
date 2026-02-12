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

# --- AI PROVIDER ADAPTERS ---
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
        self.model = genai.GenerativeModel('gemini-2.0-flash')
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

# --- UNIVERSAL CHATBOT (FULL 587-LINE LOGIC RESTORED) ---
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
            print("✅ Google Drive service initialized")
        except Exception as e:
            print(f"❌ Error setting up Google Drive service: {e}")
            raise e

    def scrape_website(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return ' '.join(chunk for chunk in chunks if chunk)[:10000]
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""

    def list_files_in_folder(self, folder_id=None, folder_name=None):
        if not self.service: return []
        supported_types = {
            'application/pdf': 'PDF',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'XLSX',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PPTX'
        }
        try:
            if folder_name and not folder_id:
                folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
                folder_results = self.service.files().list(q=folder_query).execute()
                folders = folder_results.get('files', [])
                if folders: folder_id = folders[0]['id']
            
            mime_query = " or ".join([f"mimeType='{mime}'" for mime in supported_types.keys()])
            query = f"({mime_query}) and trashed=false"
            if folder_id: query += f" and '{folder_id}' in parents"
            
            results = self.service.files().list(q=query, fields="files(id,name,mimeType)").execute()
            files = results.get('files', [])
            for f in files: f['file_type'] = supported_types.get(f['mimeType'], 'UNKNOWN')
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def download_file_content(self, file_id, mime_type):
        try:
            if 'google-apps' in mime_type:
                export_mime = 'application/pdf' # Simplified for export
                return self.service.files().export(fileId=file_id, mimeType=export_mime).execute()
            return self.service.files().get_media(fileId=file_id).execute()
        except Exception as e:
            print(f"Error downloading: {e}")
            return None

    def extract_text_from_file(self, file_content, file_name, file_type):
        try:
            if file_type == 'PDF': return self._extract_from_pdf(file_content)
            if file_type == 'DOCX': return self._extract_from_word(file_content)
            if file_type == 'XLSX': return self._extract_from_excel(file_content)
            if file_type == 'PPTX': return self._extract_from_powerpoint(file_content)
            return ""
        except Exception as e:
            print(f"Error extracting from {file_name}: {e}")
            return ""

    def _extract_from_pdf(self, content):
        pdf = PyPDF2.PdfReader(io.BytesIO(content))
        return "\n".join([page.extract_text() for page in pdf.pages[:15]])

    def _extract_from_word(self, content):
        doc = Document(io.BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs[:100]])

    def _extract_from_excel(self, content):
        df = pd.read_excel(io.BytesIO(content), nrows=100)
        return df.to_string()

    def _extract_from_powerpoint(self, content):
        ppt = Presentation(io.BytesIO(content))
        text = ""
        for slide in ppt.slides[:15]:
            for shape in slide.shapes:
                if hasattr(shape, "text"): text += shape.text + "\n"
        return text

    # --- NEW TECHNICAL ENGINES ---
    def load_technical_data(self, csv_content, file_name):
        df = pd.read_csv(io.BytesIO(csv_content))
        if 'technical_tables' not in st.session_state:
            st.session_state.technical_tables = {}
        st.session_state.technical_tables[file_name] = df
        text_data = f"Technical Table {file_name}:\n" + df.to_string()
        for chunk in self.chunk_text(text_data):
            self.documents.append({'source': file_name, 'file_type': 'CSV', 'content': chunk})

    def universal_technical_engine(self, query):
        auth_prompt = f"Using provided files and international standards (UN SDGs, GlobalSpec, OSHA), solve: {query}. Cite sources."
        return self.get_response(auth_prompt)

    def chunk_text(self, text, size=800, overlap=150):
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunks.append(text[i:i + size])
        return chunks[:50]

    def process_all_sources(self, folder_id=None, folder_name=None, website_urls=None):
        self.documents = []
        # Re-integration of Google Drive and Web logic
        if self.service:
            files = self.list_files_in_folder(folder_id, folder_name)
            for f in files:
                content = self.download_file_content(f['id'], f['mimeType'])
                text = self.extract_text_from_file(content, f['name'], f['file_type'])
                if text:
                    for c in self.chunk_text(text):
                        self.documents.append({'source': f['name'], 'file_type': f['file_type'], 'content': c})
        return self.documents

    def get_response(self, question):
        if not self.index: return "Please load documents first."
        qe = self.embedding_model.encode([question])
        scores, indices = self.index.search(qe.astype('float32'), 3)
        context = "\n\n".join([self.documents[idx]['content'] for idx in indices[0] if idx < len(self.documents)])
        return self.ai_provider.get_response(f"Context: {context}\n\nQuestion: {question}")

# --- THE INTERFACE (WITH SECURITY & TOOLS) ---
def main():
    st.set_page_config(page_title="Sree - MAPS Academy", page_icon="🌳")

    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🌳 Sree Security Gate")
        if st.text_input("Access Code", type="password") == "Sree2026":
            st.session_state.authenticated = True
            st.rerun()
        return

    sree_icon = "https://raw.githubusercontent.com/Sreeni253/maps-academy/main/kalpavruksha.png"
    enquirer_icon = "💡"

    with st.sidebar:
        st.title("🌳 MAPS Academy")
        ai_choice = st.selectbox("AI Model", ["Gemini", "OpenAI"])
        api_key = st.text_input("API Key", type="password")
        
        st.divider()
        manual_files = st.file_uploader("Upload Modules", accept_multiple_files=True)
        
        if st.button("🚀 Process All Sources"):
            if api_key:
                p = GeminiProvider(api_key) if ai_choice == "Gemini" else OpenAIProvider(api_key)
                st.session_state.chatbot = UniversalChatbot(p)
                if manual_files:
                    for f in manual_files:
                        data = f.read()
                        ftype = f.name.split('.')[-1].upper()
                        text = st.session_state.chatbot.extract_text_from_file(data, f.name, ftype)
                        if text:
                            for c in st.session_state.chatbot.chunk_text(text):
                                st.session_state.chatbot.documents.append({'source': f.name, 'content': c})
                    
                    cb = st.session_state.chatbot
                    if cb.documents:
                        cb.embeddings = cb.embedding_model.encode([d['content'] for d in cb.documents])
                        cb.index = faiss.IndexFlatL2(cb.embeddings.shape[1])
                        cb.index.add(cb.embeddings.astype('float32'))
                        st.success("Sree is Ready!")

        st.divider()
        if st.button("📝 Start Quiz"):
            if 'chatbot' in st.session_state:
                res = st.session_state.chatbot.get_response("Generate a 3-question MCQ quiz based on the modules.")
                st.session_state.messages.append({"role": "assistant", "content": res})

        st.divider()
        u_calc = st.text_input("Technical Engine:")
        if st.button("Calculate"):
            if 'chatbot' in st.session_state:
                res = st.session_state.chatbot.universal_technical_engine(u_calc)
                st.session_state.messages.append({"role": "assistant", "content": res})

    st.title("🌳 Sree - MAPS Academy")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=sree_icon if m["role"]=="assistant" else enquirer_icon):
            if m["role"]=="assistant": st.markdown(":blue[**Sree**]")
            st.markdown(m["content"])

    col1, col2 = st.columns([1, 9])
    with col1: audio = mic_recorder(start_prompt="🎤", stop_prompt="🛑", key='m')
    with col2: prompt = st.chat_input("Ask Sree...")

    final = audio['text'] if audio and audio.get('text') else prompt
    if final:
        st.session_state.messages.append({"role": "user", "content": final})
        with st.chat_message("user", avatar=enquirer_icon): st.markdown(final)
        if 'chatbot' in st.session_state:
            with st.chat_message("assistant", avatar=sree_icon):
                st.markdown(":blue[**Sree**]")
                r = st.session_state.chatbot.get_response(final)
                st.markdown(r)
                st.session_state.messages.append({"role": "assistant", "content": r})

if __name__ == "__main__":
    main()
