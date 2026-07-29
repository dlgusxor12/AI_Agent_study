# 파일명: chat_system.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

messages = [
    SystemMessage(content=(
        "너는 초등학교 5학년 학생을 가르치는 과학 선생님이야. "
        "어려운 단어는 피하고, 비유를 많이 써서 설명해 줘."
    )),
    HumanMessage(content="블랙홀이 뭐예요?"),
]

response = llm.invoke(messages)
print(response.content)
