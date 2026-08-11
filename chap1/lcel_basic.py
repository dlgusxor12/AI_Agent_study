import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_template("{topic}을 한 문장으로 설명해줘.")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

payload = {"topic": "LCEL"}

step1 = prompt.invoke(payload)
print("1단계:", type(step1), step1)

step2 = llm.invoke(step1)
print("2단계:", type(step2), step2.content)

step3 = parser.invoke(step2)
print("3단계:", type(step3), step3)

chain = prompt | llm | parser
print("한 번에:", chain.invoke(payload))
