import os
from dotenv import load_dotenv
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

class ReviewAnalysis(BaseModel):
    sentiment: Literal["긍정", "부정", "중립"] = Field(description="감정")
    score: int = Field(ge=1, le=5, description="1~5점 만족도")
    topics: List[str] = Field(description="리뷰에서 언급된 핵심 주제어")
    summary: str = Field(description="한 줄 요약")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 한 줄로 구조화 출력 LLM 생성
structured = llm.with_structured_output(ReviewAnalysis)

review = (
    "배송은 이틀 만에 왔고 포장도 깔끔했어요. "
    "다만 원단이 생각보다 얇아서 아쉬웠습니다. 그래도 디자인은 예뻐요."
)

result = structured.invoke(review)

print(type(result))
print("감정   :", result.sentiment)
print("점수   :", result.score)
print("주제어 :", result.topics)
print("요약   :", result.summary)

