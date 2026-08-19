from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/tutorials/rag/",
]
loader = WebBaseLoader(urls)
docs = loader.load()
print(f"문서 개수: {len(docs)}")  # 2