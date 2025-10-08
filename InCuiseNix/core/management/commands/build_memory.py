import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings

# --- UPDATED IMPORTS for Local Embeddings ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Command(BaseCommand):
    help = 'Builds the AI vector database using a local embeddings model.'

    def handle(self, *args, **kwargs):
        TRANSCRIPTS_DIR = os.path.join(settings.BASE_DIR.parent, 'video_outputs')
        PERSIST_DIRECTORY = os.path.join(settings.BASE_DIR.parent, "chroma_db_memory")

        all_docs = []
        self.stdout.write(self.style.NOTICE(f"[*] Reading transcripts from '{TRANSCRIPTS_DIR}'..."))
        for filename in os.listdir(TRANSCRIPTS_DIR):
            if filename.endswith("_transcript.csv"):
                file_path = os.path.join(TRANSCRIPTS_DIR, filename)
                df = pd.read_csv(file_path)
                self.stdout.write(f"  - Processing '{filename}'...")
                
                for _, row in df.iterrows():
                    text_content = str(row['Text'])
                    
                    # --- REVISED CHANGE ---
                    # The filter is now stricter, requiring more than 8 words.
                    # This will filter out most of the low-quality rhetorical questions.
                    if len(text_content.split()) > 8:
                        doc = Document(
                            page_content=text_content, 
                            metadata={'source_video': filename, 'start_time': row['Start Time'], 'end_time': row['End Time']}
                        )
                        all_docs.append(doc)

        if not all_docs:
            self.stdout.write(self.style.ERROR("[!] No transcript CSV files found or all lines were too short."))
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_docs)

        self.stdout.write(self.style.NOTICE("\n[*] Initializing Local Embeddings Model..."))
        self.stdout.write(self.style.WARNING("    (This will download a model of ~120MB the first time you run it. Please be patient.)"))

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.stdout.write(self.style.NOTICE(f"[*] Creating and saving vector database to '{PERSIST_DIRECTORY}'..."))
        Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )

        self.stdout.write(self.style.SUCCESS('\n[SUCCESS] AI memory has been built successfully using a local model!'))

