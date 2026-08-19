import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1) 로드 + 분할
docs = PyPDFLoader("./data/ICT_참여현황.pdf").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"청크 수: {len(chunks)}")

# 2) 임베딩
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3) Chroma 저장
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="acme_policy",
    persist_directory="./chroma_db",
)
print("인덱싱 완료. ./chroma_db 폴더에 저장되었습니다.")