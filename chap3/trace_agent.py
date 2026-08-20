import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """도시 이름을 받아 현재 날씨(섭씨)를 반환합니다."""
    fake_db = {"Seoul": 18, "Tokyo": 22}
    return f"{city} 현재 기온은 {fake_db.get(city, '?')}도입니다."


agent = create_agent(model="openai:gpt-5.4-mini", tools=[get_weather])

result = agent.invoke(
    {"messages": [{"role": "user", "content": "서울과 도쿄 기온 차이"}]}
)

for i, msg in enumerate(result["messages"]):
    print(f"--- [{i}] {type(msg).__name__} ---")
    if getattr(msg, "tool_calls", None):
        print("tool_calls:", msg.tool_calls)
    print("content:", msg.content)