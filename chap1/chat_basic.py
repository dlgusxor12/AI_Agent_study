# 파일명: chat_basic.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 문자열 하나를 넘기면 자동으로 HumanMessage로 감싸짐
response = llm.invoke("LangChain을 한 문장으로 설명해 주세요.")

print(type(response))       # AIMessage
print(response.content)     # 실제 답변 텍스트
