import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 한국어로 답하는 친절한 비서입니다."),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

questions = [
    "2026년 4월 오늘 날짜의 KOSPI 종가는?",
    "ACME 주식회사의 2025년 연차 규정에서 반차는 몇 시간인가요?",
    "파이썬 3.11에서 match 문의 구조는?",
]

for q in questions:
    print("=" * 60)
    print("Q:", q)
    print("A:", chain.invoke({"question": q}))