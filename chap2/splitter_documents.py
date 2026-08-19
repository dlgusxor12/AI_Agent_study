from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = PyPDFLoader("./data/ICT_참여현황.pdf").load()
print(f"원본 페이지 수: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)
chunks = splitter.split_documents(docs)
print(f"청크 수: {len(chunks)}")

print("--- 첫 청크 ---")
print(chunks[0].page_content[:200])
print(chunks[0].metadata)  # source, page 등 원본 메타데이터 그대로 유지