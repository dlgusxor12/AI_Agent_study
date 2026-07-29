# 파일명: temperature_demo.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

question = "봄을 주제로 한 줄짜리 시를 써 주세요."

for t in [0, 0.7, 1.2]:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=t,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print(f"\n=== temperature={t} ===")
    for i in range(3):
        result = llm.invoke(question)
        print(f"  {i+1}) {result.content}")
