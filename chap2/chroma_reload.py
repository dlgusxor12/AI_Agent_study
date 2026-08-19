import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 임베딩 "함수"만 필요. 실제 임베딩 호출은 검색 시 질문에만 발생
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    collection_name="acme_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

results = vectorstore.similarity_search("반차는 몇 시간?", k=3)
for i, r in enumerate(results):
    print(f"--- Top {i+1} (page {r.metadata.get('page')}) ---")
    print(r.page_content[:200])