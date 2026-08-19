import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

base = Chroma(
    collection_name="acme_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
).as_retriever(search_kwargs={"k": 6})

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base,
)

for d in retriever.invoke("반차 사용 기준"):
    print(d.page_content[:200])