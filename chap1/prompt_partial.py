import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

base = ChatPromptTemplate.from_messages([
    ("system",
     "너는 {role}이야. 항상 {language}로 답하고, 출력은 최대 {max_lines}줄."),
    ("user", "{question}"),
])

# 공통 설정을 미리 고정
korean_mentor = base.partial(language="한국어", max_lines=4)

# 이제 role과 question만 주면 됨
for role, question in [
    ("파이썬 교사", "list와 tuple의 차이가 뭐야?"),
    ("헬스 트레이너", "스쿼트 자세 주의점 알려줘"),
]:
    msgs = korean_mentor.format_messages(role=role, question=question)
    print(f"\n[{role}]")
    print(llm.invoke(msgs).content)
