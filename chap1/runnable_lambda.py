import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

def normalize(input_dict: dict) -> dict:
    """사용자 입력을 정돈: 공백 제거, 소문자화는 하지 않음(한국어 포함)."""
    return {"question": input_dict["question"].strip()}

def add_suffix(text: str) -> str:
    return f"{text} "

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "한 줄로 짧게 답해."),
    ("user", "{question}"),
])

chain = (
    RunnableLambda(normalize)
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(add_suffix)
)

# 입력에 앞뒤 공백을 일부러 넣음
result = chain.invoke({"question": "   파이썬의 가장 큰 장점은?   "})
print(result)