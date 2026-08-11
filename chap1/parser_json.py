import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "사용자 요청을 JSON으로 답해. 스키마: "
     '{{"intent": "검색|주문|문의|기타", "keywords": ["문자열 배열"]}}. '
     "다른 말 금지."),
    ("user", "{utterance}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = prompt | llm | parser

result = chain.invoke({"utterance": "주말에 입을 얇은 봄 자켓 추천해줘"})
print(type(result))
print(result)
print("의도:", result["intent"])
print("키워드:", result["keywords"])