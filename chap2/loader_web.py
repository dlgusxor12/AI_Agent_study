from langchain_community.document_loaders import WebBaseLoader

url = "https://python.langchain.com/docs/introduction/"
loader = WebBaseLoader(url)
docs = loader.load()

print(f"문서 개수: {len(docs)}")
print(f"메타데이터: {docs[0].metadata}")
print(f"본문 앞 300자:\n{docs[0].page_content[:300]}")