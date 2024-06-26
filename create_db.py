import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from azure_models import embedding

embeddings = embedding()

pdf_directory = "Documents"
for filename in os.listdir(pdf_directory):
    if not filename.endswith(".pdf"):
      continue
    filepath = os.path.join(pdf_directory, filename)
    loader = PDFMinerLoader(filepath)
    docs = loader.load()
    text_splitter = SemanticChunker(embeddings)
    texts = text_splitter.split_documents(docs)
    vector_db = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
    print(filename,"- Embedding done...")

print("Embedding done and Database created Successfully.")