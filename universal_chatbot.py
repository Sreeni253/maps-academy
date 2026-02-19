import sys
import os

# This tells your chatbot to look inside the 'backend' folder for the academy code
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
import academy_logic
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

# AI Provider Adapters (This is the ONLY part that changes!)

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
        self.model = genai.GenerativeModel('gemini-2.5-flash')

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

# Universal Chatbot (Same for ALL AI providers!)

class UniversalChatbot:
    def __init__(self, ai_provider, credentials_json=None):
        self.ai_provider = ai_provider
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Google Drive setup
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
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
    
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
    
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
    
            return text[:10000]
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""
    
    def list_files_in_folder(self, folder_id=None, folder_name=None):
        if not self.service:
            return []
    
        supported_types = {
            'application/pdf': 'PDF',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
            'application/msword': 'DOC',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'XLSX',
            'application/vnd.ms-excel': 'XLS',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'PPTX',
            'application/vnd.ms-powerpoint': 'PPT',
            'application/vnd.google-apps.document': 'Google Doc',
            'application/vnd.google-apps.spreadsheet': 'Google Sheet',
            'application/vnd.google-apps.presentation': 'Google Slides'
        }
    
        try:
            if folder_name and not folder_id:
                folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
                folder_results = self.service.files().list(q=folder_query).execute()
                folders = folder_results.get('files', [])
    
                if not folders:
                    raise Exception(f"Folder '{folder_name}' not found")
                folder_id = folders[0]['id']
                print(f"Found folder: {folder_name}")
    
            mime_query = " or ".join([f"mimeType='{mime}'" for mime in list(supported_types.keys())[:5]])
    
            if folder_id:
                query = f"({mime_query}) and '{folder_id}' in parents and trashed=false"
            else:
                query = f"({mime_query}) and trashed=false"
    
            results = self.service.files().list(
                q=query,
                fields="files(id,name,mimeType,size,modifiedTime)",
                pageSize=15
            ).execute()
    
            files = results.get('files', [])
            for file in files:
                file['file_type'] = supported_types.get(file['mimeType'], 'UNKNOWN')
    
            return files
    
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def download_file_content(self, file_id, mime_type):
        try:
            if mime_type == 'application/vnd.google-apps.document':
                request = self.service.files().export(fileId=file_id, 
                    mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                request = self.service.files().export(fileId=file_id, 
                    mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            elif mime_type == 'application/vnd.google-apps.presentation':
                request = self.service.files().export(fileId=file_id, 
                    mimeType='application/vnd.openxmlformats-officedocument.presentationml.presentation')
            else:
                request = self.service.files().get_media(fileId=file_id)
    
            return request.execute()
        except Exception as e:
            print(f"Error downloading file {file_id}: {e}")
            return None
    
    def extract_text_from_file(self, file_content, file_name, file_type):
        try:
            if file_type in ['PDF']:
                return self._extract_from_pdf(file_content)
            elif file_type in ['DOCX', 'DOC', 'Google Doc']:
                return self._extract_from_word(file_content)
            elif file_type in ['XLSX', 'XLS', 'Google Sheet']:
                return self._extract_from_excel(file_content)
            elif file_type in ['PPTX', 'PPT', 'Google Slides']:
                return self._extract_from_powerpoint(file_content)
            else:
                return ""
        except Exception as e:
            print(f"Error extracting from {file_name}: {e}")
            return ""
    
    def _extract_from_pdf(self, pdf_content):
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages[:10]:
            text += page.extract_text() + "\n"
        return text
    
    def _extract_from_word(self, docx_content):
        doc_file = io.BytesIO(docx_content)
        doc = Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs[:50]:
            text += paragraph.text + "\n"
        return text
    
    def _extract_from_excel(self, excel_content):
        excel_file = io.BytesIO(excel_content)
        try:
            excel_data = pd.read_excel(excel_file, sheet_name=None, nrows=100)
            text = ""
            for sheet_name, df in excel_data.items():
                text += f"\n--- Sheet: {sheet_name} ---\n"
                text += df.head(20).to_string(index=False) + "\n"
            return text
        except Exception as e:
            return ""
    
    def _extract_from_powerpoint(self, pptx_content):
        ppt_file = io.BytesIO(pptx_content)
        presentation = Presentation(ppt_file)
        text = ""
        for slide_num, slide in enumerate(presentation.slides[:10], 1):
            text += f"\n--- Slide {slide_num} ---\n"
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
        return text
    
    def load_technical_data(self, content, name):
        """Processes CSV and PDF technical tables for the Universal AI Chatbot"""
        try:
            ext = name.split('.')[-1].lower()
            if ext == 'csv':
                df = pd.read_csv(io.BytesIO(content))
                text_data = f"Technical Table (CSV) {name}:\n" + df.to_string()
            elif ext == 'pdf':
                # Labels it correctly for the Universal AI Chatbot logic
                text_data = f"Technical Table (PDF Extract) {name}:\n" + self._extract_from_pdf(content)
    
            # Store it so the Universal AI Chatbot knows this is "Authority Data"
            for chunk in self.chunk_text(text_data):
                self.documents.append({
                    'source': name, 
                    'file_type': 'Technical Table', 
                    'content': chunk
                })
            print(f"    ✅ Technical data loaded into Universal AI Chatbot from {name}")
        except Exception as e:
            print(f"    ❌ Data Load Error: {e}")
    
    def universal_technical_engine(self, query):
        """The brain that connects your data to Global Standards"""
        authority_prompt = f"Using technical files and standards (UN SDGs, GlobalSpec, OSHA), solve: {query}"
        return self.get_response(authority_prompt)
    
    def chunk_text(self, text, chunk_size=800, overlap=150):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks[:20]
    
    def process_all_sources(self, folder_id=None, folder_name=None, website_urls=None):
        print("🔍 Processing all sources...")
        self.documents = []
        processed_files = []
        processed_sites = []
    
      # --- 1. Process Google Drive files
        if self.service:
            files = self.list_files_in_folder(folder_id, folder_name)
    
            if files:
                files = files[:10]
                print(f"📁 Processing {len(files)} files from Google Drive")
    
                for file_info in files:
                    file_id = file_info['id']
                    file_name = file_info['name']
                    mime_type = file_info['mimeType']
                    file_type = file_info['file_type']
    
                    print(f"📄 Processing {file_type}: {file_name}")
    
                    try:
                        file_content = self.download_file_content(file_id, mime_type)
    
                        if file_content:
                            text = self.extract_text_from_file(file_content, file_name, file_type)
    
                            if text.strip():
                                chunks = self.chunk_text(text)
    
                                for i, chunk in enumerate(chunks):
                                    self.documents.append({
                                        'source': file_name,
                                        'file_type': file_type,
                                        'content': chunk
                                    })
    
                                processed_files.append({
                                    'name': file_name,
                                    'type': file_type,
                                    'chunks': len(chunks)
                                })
                                print(f"    ✅ Extracted {len(chunks)} chunks")
                            else:
                                print(f"    ⚠️  No text extracted")
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                        continue
    
       # --- 2. Process Websites ---
        if website_urls:
            urls = [url.strip() for url in website_urls.split('\n') if url.strip()]
            if urls:
                urls = urls[:5]
                print(f"🌐 Processing {len(urls)} websites")
                for url in urls:
                    print(f"🌐 Scraping: {url}")
                    try:
                        text = self.scrape_website(url)
                        if text.strip():
                            chunks = self.chunk_text(text)
                            for chunk in chunks:
                                self.documents.append({
                                    'source': url,
                                    'file_type': 'Website',
                                    'content': chunk
                                })
                            processed_sites.append({'name': url, 'type': 'Website', 'chunks': len(chunks)})
                            print(f"    ✅ Extracted {len(chunks)} chunks")
                            time.sleep(1) # Polite delay
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                        continue
    
        # --- 3. Finalize Knowledge Base ---
        all_processed = processed_files + processed_sites
        if not self.documents:
            raise ValueError("No content could be extracted from any sources")
    
        print(f"\n📊 Processing Summary:")
        print(f"   • Files processed: {len(processed_files)}")
        print(f"   • Websites processed: {len(processed_sites)}")
        print(f"   • Total text chunks: {len(self.documents)}")
    
        print("\n🔄 Creating embeddings...")
        all_chunks = [doc['content'] for doc in self.documents]
        self.embeddings = self.embedding_model.encode(all_chunks)
    
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
        print("✅ Ready to chat!")
        return all_processed
    
    def search_similar_chunks(self, query, k=3):
        if self.index is None: return []
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({'content': self.documents[idx]['content'], 'source': self.documents[idx]['source'], 'file_type': self.documents[idx]['file_type']})
        return results
    
    def get_response(self, question):
        """The ONLY method that uses AI provider - everything else is universal!"""
        if not self.index:
            return "Please load your training modules in the sidebar first."
    
        # 1. Search for relevant technical context
        relevant_chunks = self.search_similar_chunks(question, k=3)
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        sources = list(set([f"{chunk['source']} ({chunk['file_type']})" for chunk in relevant_chunks]))
    
        # 2. Create the Engineering-Focused Prompt
        prompt = f"""Answer the user's question using the information from these sources. 
        If the question involves a calculation and the formula is in the sources, perform the calculation.
        Be direct, professional, and natural.

Sources:
{context}

Question: {question}

Answer naturally:"""

        try:
            # THIS is the ONLY line that changes between AI providers!
            answer = self.ai_provider.get_response(prompt)
    
            if sources:
                answer += f"\n\n**Sources:** {', '.join(sources)}"
    
            return answer
    
        except Exception as e:
            return f"Error getting response from AI: {str(e)}"

# Streamlit Interface (Universal for ALL AI providers!)

def main():
    # Keep your original branding
    st.set_page_config(page_title="Universal AI Chatbot", page_icon="🤖")

    # --- SECURITY GATE START ---
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🤖 Universal AI Chatbot")
        st.markdown("### 🔐 Secure Access")
    
        # Password Input
        access_code = st.text_input("Enter Access Code:", type="password")
    
        if st.button("Unlock Chatbot"):
            if access_code == "Sree2026":
                st.session_state.authenticated = True
                st.success("Access Granted!")
                st.rerun() 
            else:
                st.error("Incorrect code.")
        return # Security gate
    # --- SECURITY GATE END ---
    
    st.title("🤖 Universal AI Chatbot")
    st.markdown("**Works with OpenAI, Gemini, Claude, or any AI provider!**")

    # --- THE ACADEMY ENGINE ---
    selected_step = st.session_state.get('academy_step')
    
    if "chatbot" in st.session_state and hasattr(st.session_state.chatbot, 'documents') and st.session_state.chatbot.documents:
        manual_content = "\n".join([doc['content'] for doc in st.session_state.chatbot.documents])
    else:
        manual_content = st.session_state.get('manual_text', '')
    
    if selected_step:
        import academy_logic
        academy_logic.execute_academy_step(selected_step, manual_content)
        st.divider()
    else:
        st.info("Select a Step from the 'Skill Validation' sidebar to begin your Maps Academy training.")
        st.markdown("**Works with OpenAI, Gemini, Claude, or any AI provider!**")

# Sidebar configuration

    with st.sidebar:
        st.header("🧠 Choose Your AI")
    
        ai_choice = st.selectbox(
            "Select AI Provider:",
            ["Gemini (Free)", "OpenAI (Paid)", "Claude (Paid)"]
        )
    
        # Get API key based on choice
        if ai_choice.startswith("Gemini"):
            api_key = st.text_input("Gemini API Key:", type="password")
            st.caption("Get FREE key from https://makersuite.google.com/app/apikey")
        elif ai_choice.startswith("OpenAI"):
            api_key = st.text_input("OpenAI API Key:", type="password")
            st.caption("Get key from https://platform.openai.com")
        else:  # Claude
            api_key = st.text_input("Claude API Key:", type="password")
            st.caption("Get key from https://console.anthropic.com")
    
        # --- NEW: MAPS ACADEMY CONTROLS INSERTED HERE ---
        st.divider()
        st.header("🎓 Maps Academy")
    
        if st.button("📊 Step 1: Generate Presentation"):
            st.session_state['academy_step'] = "Step 1: Fixed Presentation"
            st.rerun()
    
        if st.button("👨‍🏫 Step 2: Open Tutor Mode"):
            st.session_state['academy_step'] = "Step 2: The Tutor"
            st.rerun()
    
        if st.button("🎓 Step 3: Generate Graduation Quiz"):
            st.session_state['academy_step'] = "Step 3: Graduation Quiz"
            st.rerun()
    
        if st.button("💬 Return to Chat"):
            st.session_state['academy_step'] = None
            st.rerun()
        st.divider()
        # -----------------------------------------------
    
        st.header("📁 Google Drive Setup")
        uploaded_json = st.file_uploader("Upload Google Service Account JSON", type="json")
    
        folder_option = st.radio(
            "Google Drive files:",
            ["Skip Drive", "Specific Folder by ID", "Specific Folder by Name"]
        )
    
        folder_id = None
        folder_name = None
        if folder_option == "Specific Folder by Name":
            folder_name = st.text_input("Folder Name:")
        elif folder_option == "Specific Folder by ID":
            folder_id = st.text_input("Folder ID:")
    
        st.header("🌐 Website Sources")
        website_urls = st.text_area("Website URLs (one per line):", height=100)
    
       # --- NEW: UNIVERSAL FILE UPLOADER SECTION ---
        st.header("📤 Local Training Modules")
        manual_files = st.file_uploader(
            "Upload PDF, Word, or Markdown files:", 
            type=None, 
            accept_multiple_files=True
        )
    
        if st.button("🚀 Process All Sources"):
            if api_key:
                try:
                    with st.spinner(f"Processing with {ai_choice}..."):
                        if ai_choice.startswith("Gemini"):
                            ai_provider = GeminiProvider(api_key)
                        elif ai_choice.startswith("OpenAI"):
                            ai_provider = OpenAIProvider(api_key)
                        else:  # Claude
                            ai_provider = ClaudeProvider(api_key)
    
                        credentials_json = None
                        if uploaded_json:
                            credentials_json = json.loads(uploaded_json.getvalue().decode())
    
                        chatbot = UniversalChatbot(ai_provider, credentials_json)
                        st.session_state.chatbot = chatbot
    
                        all_processed = []
                        try:
                            drive_web_sources = chatbot.process_all_sources(
                                folder_id=folder_id, 
                                folder_name=folder_name, 
                                website_urls=website_urls
                            )
                            all_processed.extend(drive_web_sources)
                        except:
                            pass 
    
                        if manual_files:
                            for uploaded_file in manual_files:
                                file_content = uploaded_file.read()
                                file_name = uploaded_file.name
    
                                text = ""
                                if file_name.endswith('.pdf'):
                                    text = chatbot._extract_from_pdf(file_content)
                                    chatbot.load_technical_data(file_content, file_name)
                                elif file_name.endswith('.docx'):
                                    text = chatbot._extract_from_word(file_content)
                                elif file_name.endswith(('.txt', '.md')):
                                    text = file_content.decode('utf-8')
    
                                if text.strip():
                                    chunks = chatbot.chunk_text(text)
                                    for chunk in chunks:
                                        chatbot.documents.append({
                                            'source': file_name,
                                            'file_type': file_name.split('.')[-1].upper(),
                                            'content': chunk
                                        })
                                    all_processed.append({
                                        'name': file_name,
                                        'type': file_name.split('.')[-1].upper(),
                                        'chunks': len(chunks)
                                    })
    
                        if chatbot.documents:
                            all_chunks = [doc['content'] for doc in chatbot.documents]
                            chatbot.embeddings = chatbot.embedding_model.encode(all_chunks)
                            dimension = chatbot.embeddings.shape[1]
                            chatbot.index = faiss.IndexFlatL2(dimension)
                            chatbot.index.add(chatbot.embeddings.astype('float32'))
    
                            st.session_state.processed_sources = all_processed
                            st.session_state['manual_text'] = "\n".join([doc['content'] for doc in chatbot.documents])
    
                            st.success(f"✅ Ready! Loaded {len(all_processed)} sources.")
    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.error("Please provide API key")
    
        if hasattr(st.session_state, 'processed_sources'):
            st.subheader("📚 Loaded Sources")
            for source_info in st.session_state.processed_sources:
                source_name = source_info['name']
                if len(source_name) > 30:
                    source_name = source_name[:27] + "..."
                st.text(f"✅ {source_name} ({source_info['chunks']} chunks)")

# --- BRANDED CHAT INTERFACE WITH VOICE ---

    from streamlit_mic_recorder import mic_recorder
    
    sree_icon = "https://raw.githubusercontent.com/Sreeni253/maps-academy/main/kalpavruksha.png"
    enquirer_icon = "💡" 
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 1. Display conversation history
    for message in st.session_state.messages:
        avatar = sree_icon if message["role"] == "assistant" else enquirer_icon
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                st.markdown(":blue[**Sree**]") 
            st.markdown(message["content"])
    
    # 2. Voice and Text Input Section
    footer_col1, footer_col2 = st.columns([1, 9])
    
    with footer_col1:
        audio = mic_recorder(start_prompt="🎤", stop_prompt="🛑", key='sree_mic')
    
    with footer_col2:
        prompt = st.chat_input("Speak with Sree...")
    
    # 3. Process Input
    final_prompt = None
    if audio and audio.get('text'):
        final_prompt = audio['text']
    elif prompt:
        final_prompt = prompt
    
    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user", avatar=enquirer_icon):
            st.markdown(final_prompt)
    
        if "chatbot" in st.session_state and st.session_state.chatbot is not None:
            try:
                with st.chat_message("assistant", avatar=sree_icon):
                    st.markdown(":blue[**Sree**]")
                    with st.spinner("Sree is consulting the training modules..."):
                        response = st.session_state.chatbot.ask_question(final_prompt)
                        st.markdown(response)
                        
                        # --- THE CRITICAL DATA BRIDGE ---
                        # This forces the presentation to save regardless of source
                        if st.session_state.get('academy_step') == "Step 1: Fixed Presentation":
                            st.session_state.current_presentation = response
                            
                        # Update manual_text for Step 2 and Step 3
                        if hasattr(st.session_state.chatbot, 'documents'):
                            st.session_state['manual_text'] = "\n".join([doc['content'] for doc in st.session_state.chatbot.documents])
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"❌ Logic Error: {str(e)}")
        else:
            st.warning("⚠️ Please upload your manual and click '🚀 Process All Sources' first!")
    
        if hasattr(st.session_state, 'chatbot'):
            with st.chat_message("assistant", avatar=sree_icon):
                st.markdown(":blue[**Sree**]")
                with st.spinner("Sree is consulting the training modules..."):
                    response = st.session_state.chatbot.ask_question(final_prompt)
                    st.markdown(response)
                    
                    # --- ADD THIS LINE TO SAVE THE PRESENTATION ---
                    if st.session_state.get('academy_step') == "Step 1: Fixed Presentation":
                        st.session_state.current_presentation = response
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
        st.sidebar.divider()
        st.sidebar.subheader("🎓 Skill Validation")
        if st.sidebar.button("📝 Generate Graduation Quiz"):
            if "chatbot" in st.session_state and st.session_state.chatbot.documents:
                with st.sidebar:
                    with st.spinner("Sree is preparing your exam..."):
                        quiz_query = (
                            "You are an expert instructor for MAPS Academy. "
                            "Based on the technical documents provided, generate a 3-question "
                            "Multiple Choice Quiz. Provide the questions first, "
                            "followed by an 'Answer Key' section at the bottom."
                            )
                        quiz_response = st.session_state.chatbot.get_response(quiz_query)
                        st.session_state.current_quiz = quiz_response
            else:
                st.sidebar.warning("Please upload and process files first!")
    
    if "current_quiz" in st.session_state:
        st.markdown("---")
        st.subheader("🎓 Sree's Graduation Quiz")
        st.write(st.session_state.current_quiz)
    
        if st.button("🗑️ Clear Quiz and Return to Chat", key="final_close_btn"):
            del st.session_state.current_quiz
            st.rerun()

if __name__ == "__main__":
    main()
