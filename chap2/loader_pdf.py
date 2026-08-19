from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./data/ICT_참여현황.pdf")
docs = loader.load()

print(f"총 페이지 수: {len(docs)}")
print(f"첫 페이지 내용 앞부분:\n{docs[0].page_content[:200]}")
print(f"첫 페이지 메타데이터: {docs[0].metadata}")
