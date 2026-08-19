from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
ACME 연차 규정 제1조 (목적)
이 규정은 회사 임직원의 연차휴가 사용 및 관리에 관한 사항을 정함을 목적으로 한다.

제2조 (적용 범위)
본 규정은 회사 소속 모든 정규직 임직원에게 적용된다. 단, 계약직·인턴은 별도의 규정을 따른다.

제3조 (반차)
반차는 4시간으로 하며, 오전 반차와 오후 반차로 구분한다. 반차 사용 시 점심시간은 포함하지 않는다.
""" * 5  # 일부러 길게

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
chunks = splitter.split_text(text)

print(f"청크 수: {len(chunks)}")
for i, c in enumerate(chunks[:3]):
    print(f"--- chunk {i} (len={len(c)}) ---")
    print(c)