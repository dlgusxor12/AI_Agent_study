# 파일명: count_tokens.py
import tiktoken

# gpt-4o 계열은 o200k_base 토크나이저를 사용
enc = tiktoken.encoding_for_model("gpt-4o-mini")

samples = [
    "Hello, world!",
    "안녕하세요, 반갑습니다.",
    "LangChain으로 에이전트 만들기",
    "The quick brown fox jumps over the lazy dog.",
]

for text in samples:
    tokens = enc.encode(text)
    print(f"[{len(tokens):>3} 토큰] {text}")
    print(f"         조각: {[enc.decode([t]) for t in tokens]}")
    print()
