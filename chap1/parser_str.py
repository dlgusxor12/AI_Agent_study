import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "한 문장으로 정의해."),
    ("user", "{term}"),
])

# 파서 없을 때
raw_chain = prompt | llm
raw_result = raw_chain.invoke({"term": "벡터 데이터베이스"})
print("=== 파서 없을 때 ===")
print(type(raw_result), "→", raw_result)

# 파서 있을 때
chain = prompt | llm | StrOutputParser()
text = chain.invoke({"term": "벡터 데이터베이스"})
print("\n=== StrOutputParser 사용 ===")
print(type(text), "→", text)