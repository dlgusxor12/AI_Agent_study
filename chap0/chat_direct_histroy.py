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

history = [
    {"role": "system", "content": "너는 한국어로 답하는 친절한 비서다."},
    {"role": "user",   "content": "파이썬을 한 줄로 소개해 줘."},
]

# 1차 호출
r1 = client.chat.completions.create(model="gpt-4o-mini", messages=history, temperature=0)
answer1 = r1.choices[0].message.content
print("A1:", answer1)

# 모델의 답을 history 에 추가하고, 후속 질문을 던진다
history.append({"role": "assistant", "content": answer1})
history.append({"role": "user",      "content": "그럼 자바스크립트는?"})

# 2차 호출
r2 = client.chat.completions.create(model="gpt-4o-mini", messages=history, temperature=0)
print("A2:", r2.choices[0].message.content)

