import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

question = "반차는 몇 시간인가요?"
candidates = [
    "반차는 4시간이며 오전/오후로 나눈다.",
    "연차는 입사 1년 후 15일이 부여된다.",
    "점심시간은 12시부터 13시까지이다.",
    "오늘 회의는 오후 3시입니다.",
]

q_vec = embeddings.embed_query(question)
d_vecs = embeddings.embed_documents(candidates)

scored = [(cosine(q_vec, d_vec), text) for d_vec, text in zip(d_vecs, candidates)]
scored.sort(reverse=True)

print(f"질문: {question}\n")
for score, text in scored:
    print(f"  {score:.3f} | {text}")