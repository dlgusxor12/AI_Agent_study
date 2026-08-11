import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# 1) 원하는 스키마 정의
class MeetingSummary(BaseModel):
    topic: str = Field(description="회의의 주된 주제")
    attendees: List[str] = Field(description="참석자 이름 리스트")
    decisions: List[str] = Field(description="합의/결정된 사항 리스트")
    next_actions: List[str] = Field(description="후속 액션 아이템")

# 2) 파서 생성
parser = PydanticOutputParser(pydantic_object=MeetingSummary)

# 3) 프롬프트에 스키마 지시문 자동 주입
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "다음 회의록을 구조화해서 JSON으로 출력해.\n"
     "{format_instructions}"),
    ("user", "{minutes}"),
]).partial(format_instructions=parser.get_format_instructions())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = prompt | llm | parser

minutes = """
[2025-04-21 마케팅 주간 회의]
참석: 김지민(팀장), 박서연, 이도윤
- Q2 캠페인 타겟 연령대 논의 → 20~30대 여성으로 확정
- 광고 예산 15% 증액 건 승인
- 이도윤이 경쟁사 리서치 자료 금주 내로 공유
- 다음 주부터 매주 화요일 오후 3시 정기 회의로 변경
"""

result = chain.invoke({"minutes": minutes})

print(type(result))
print("주제       :", result.topic)
print("참석자     :", result.attendees)
print("결정 사항  :", result.decisions)
print("후속 액션  :", result.next_actions)