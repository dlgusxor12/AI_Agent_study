import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vs = Chroma(
    collection_name="acme_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

results = retriever.invoke("반차는 몇 시간인가요?")
print(f"검색 결과: {len(results)}개\n")
for i, d in enumerate(results):
    page = d.metadata.get("page")
    print(f"--- [{i+1}] page {page} ---")
    print(d.page_content[:160])