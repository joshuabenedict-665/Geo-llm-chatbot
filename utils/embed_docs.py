from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

doc_dir = "docs/whitebox"  # Or qgis, gdal

documents = []
for filename in os.listdir(doc_dir):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(doc_dir, filename))
        documents.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding)
db.save_local("vector_store/tools")
