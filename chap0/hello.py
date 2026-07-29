# 파일명: hello.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1) .env 에서 환경변수 로드
load_dotenv()

# 2) 키 존재 점검 (없으면 친절하게 실패)
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 3) LLM 인스턴스 생성
llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
)

# 4) 질문 한 번 보내고 응답 출력
response = llm.invoke("한 문장으로 자기 소개해줘.")
print(response.content)
