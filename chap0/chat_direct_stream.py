# 파일명: chat_direct.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1) .env 에서 환경변수 로드
load_dotenv()

# 2) 키 존재 점검 (없으면 친절하게 실패)
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 3) 클라이언트 생성 — api_key 인자를 안 주면 환경변수를 자동으로 사용
client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    stream=True,
    messages=[
        {"role": "user", "content": "파이썬으로 별 다섯 줄 그려 줘."},
    ],
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
