import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "여기 1,000자 정도의 샘플 텍스트..." * 50   # 대략 5만 자
tokens = len(enc.encode(text))
print(f"대략 토큰 수: {tokens}")
# 간단 환산 예: 1천 토큰 ≈ $0.00002 (text-embedding-3-small 기준, 시점에 따라 변동)