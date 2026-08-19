from langchain_text_splitters import MarkdownHeaderTextSplitter

md = """# 프로젝트 개요
본 프로젝트는 ...

## 설치
pip install foo

## 사용법
from foo import bar
"""

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
    ]
)
chunks = splitter.split_text(md)

for c in chunks:
    print(c.metadata, "|", c.page_content[:40])