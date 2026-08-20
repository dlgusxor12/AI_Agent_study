import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """도시 이름을 받아 현재 날씨 문자열을 반환합니다."""
    return f"{city}의 날씨: 맑음, 20도"


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("서울 날씨 알려줘")
print("content:", repr(response.content))
print("tool_calls:", response.tool_calls)