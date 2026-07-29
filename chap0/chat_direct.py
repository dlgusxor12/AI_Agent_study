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

# 4) 한 번 호출
response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.7,
    messages=[
        {"role": "system", "content": "너는 한국어로 답하는 친절한 비서다."},
        {"role": "user", "content": "한 문장으로 자기 소개해줘."},
    ],
)

# 5) 응답 텍스트 출력
# print(response.choices[0].message.content)

choice = response.choices[0]
usage = response.usage

print("==== 응답 ====")
print(choice.message.content)
print()
print("==== 메타 ====")
print("model         :", response.model)
print("finish_reason :", choice.finish_reason)
print("prompt 토큰   :", usage.prompt_tokens)
print("completion 토큰:", usage.completion_tokens)
print("total 토큰    :", usage.total_tokens)
