import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 감정 분석가야. 리뷰가 긍정/부정/중립 중 무엇인지 한 단어로만 답해."),
    ("user", "{review}"),
])

# 체인 구성
chain = prompt | llm | parser

# dict 하나로 호출!
reviews = [
    "배송이 너무 느렸어요. 다시는 안 삽니다.",
    "기대 이상이에요. 재구매 의사 있습니다.",
    "그냥 그래요. 특별히 나쁘진 않아요.",
]

for r in reviews:
    label = chain.invoke({"review": r})
    print(f"[{label:>3}] {r}")
