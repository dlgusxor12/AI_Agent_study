import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "짧게 답해."),
    ("user", "{country}의 수도는?"),
])
chain = prompt | ChatOpenAI(model="gpt-5.4-mini", temperature=0) | StrOutputParser()

# 1) invoke — 하나 실행
print("=== invoke ===")
print(chain.invoke({"country": "프랑스"}))

# 2) batch — 여러 개 병렬
print("\n=== batch ===")
print(chain.batch([
    {"country": "독일"},
    {"country": "일본"},
    {"country": "브라질"},
]))

# 3) stream — 한 글자씩 흘려보기
print("\n=== stream ===")
for chunk in chain.stream({"country": "이탈리아"}):
    print(chunk, end="", flush=True)
print()

# 4) astream — 비동기 스트림
async def run_async():
    print("=== astream ===")
    async for chunk in chain.astream({"country": "캐나다"}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(run_async())