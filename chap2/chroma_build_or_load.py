import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = "./chroma_db"
COLLECTION = "acme_policy"


def build_or_load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 이미 인덱스 파일이 있으면 재로드
    if os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite3")):
        print("[INFO] 기존 인덱스 재로드")
        return Chroma(
            collection_name=COLLECTION,
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )

    # 없으면 새로 만들기
    print("[INFO] 새 인덱스 생성")
    docs = PyPDFLoader("./data/sample.pdf").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )


if __name__ == "__main__":
    vs = build_or_load_vectorstore()
    print(vs.similarity_search("반차는 몇 시간?", k=1)[0].page_content[:120])