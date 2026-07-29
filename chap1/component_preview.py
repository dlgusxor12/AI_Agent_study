# 파일명: components_preview.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# [1] Models
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# [2] Prompts
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 한국어 IT 기자야. 전문 용어는 쉬운 말로 풀어줘."),
    ("user", "{topic}에 대해 3문장으로 요약해 줘."),
])

# [3] Output Parsers
parser = StrOutputParser()

# [6] Chains (LCEL) — 파이프로 연결!
chain = prompt | llm | parser

# 실행
result = chain.invoke({"topic": "쿠버네티스"})
print(result)
