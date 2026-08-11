# 파일명: lcel_print_graph.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 영화 추천가야."),
    ("user", "{mood}한 기분일 때 볼 만한 영화 3편을 추천해줘."),
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
parser = StrOutputParser()

chain = prompt | llm | parser

# 그래프 출력
chain.get_graph().print_ascii()
