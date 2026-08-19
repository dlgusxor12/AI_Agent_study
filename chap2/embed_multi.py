import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = [
    "반차는 4시간이며 오전/오후로 나눈다.",
    "연차는 입사 1년 후 15일이 부여된다.",
    "점심시간은 12시부터 13시까지이다.",
]

vectors = embeddings.embed_documents(texts)
print(f"입력 문서 수: {len(texts)}")
print(f"반환된 벡터 수: {len(vectors)}")
print(f"각 벡터 차원: {len(vectors[0])}")