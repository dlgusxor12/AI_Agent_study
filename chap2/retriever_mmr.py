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
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.5,
    },
)

for d in retriever.invoke("반차와 연차와 반반차의 차이는?"):
    print(d.metadata.get("page"), "|", d.page_content[:80])