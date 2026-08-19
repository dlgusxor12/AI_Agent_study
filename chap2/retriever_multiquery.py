import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

base = Chroma(
    collection_name="acme_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
).as_retriever(search_kwargs={"k": 4})

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=base,
    llm=llm,
)

# LangChain 내부 로그로 어떤 변형이 생성됐는지 보고 싶을 때
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

results = mq_retriever.invoke("반차 문의")
print(f"합쳐진 결과: {len(results)}개")
for d in results[:4]:
    print("-", d.page_content[:80])