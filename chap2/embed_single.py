import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vec = embeddings.embed_query("반차는 몇 시간인가요?")
print(f"벡터 차원: {len(vec)}")        # 1536
print(f"앞 5개 값: {vec[:5]}")