import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vs = Chroma(
    collection_name="acme_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# 추가
new_docs = [
    Document(
        page_content="제10조 (원격근무) 재택근무는 주 2회까지 허용한다.",
        metadata={"source": "update_2026-04.md", "page": 0},
    ),
]
ids = vs.add_documents(new_docs)
print("추가된 ID:", ids)

# 확인
print(vs.similarity_search("재택근무는 며칠까지?", k=1)[0].page_content)

# 삭제 (필요 시)
# vs.delete(ids=ids)

## db 삭제
# rmdir /s /q chroma_db

## DB 삭제 (python)
# import shutil
# shutil.rmtree("./chroma_db", ignore_errors=True)