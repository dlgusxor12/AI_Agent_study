from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/examplefile.txt", encoding="utf-8")
docs = loader.load()

print(f"문서 개수: {len(docs)}")          # 1
print(f"메타데이터: {docs[0].metadata}")   # {'source': './data/examplefile.txt'}
print(f"본문 앞 100자:\n{docs[0].page_content[:100]}")