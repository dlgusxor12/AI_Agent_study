# 파일명: lcel_minimal.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 여행 가이드야."),
    ("user", "{city}에서 하루 동안 꼭 해봐야 할 일 3가지를 번호와 함께 알려줘."),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()

# 체인 구성 — 이 한 줄이 LCEL의 전부
chain = prompt | llm | parser

# 실행
result = chain.invoke({"city": "교토"})
print(result)

# 스트리밍도 같은 체인에서 그대로 가능
print("\n=== 스트리밍 ===")
for token in chain.stream({"city": "부산"}):
    print(token, end="", flush=True)
print()
