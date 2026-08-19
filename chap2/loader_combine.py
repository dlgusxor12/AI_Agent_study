from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

pdf_docs = PyPDFLoader("./data/ICT_참여현황.pdf").load()
web_docs = WebBaseLoader("https://python.langchain.com/docs/introduction/").load()

all_docs = pdf_docs + web_docs
print(f"총 문서 수: {len(all_docs)}")

# 메타데이터 소스를 통일된 형식으로 가공
for d in all_docs:
    src = d.metadata.get("source", "unknown")
    d.metadata["source_type"] = "pdf" if src.endswith(".pdf") else "web"

print(all_docs[0].metadata)
print(all_docs[-1].metadata)