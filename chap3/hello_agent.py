import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """도시 이름을 받아 현재 날씨(섭씨)를 반환합니다.

    Args:
        city: 도시 이름(예: "Seoul", "Tokyo")
    """
    fake_db = {"Seoul": 18, "Tokyo": 22, "Busan": 20}
    temp = fake_db.get(city)
    if temp is None:
        return f"{city}의 날씨 정보를 찾을 수 없습니다."
    return f"{city} 현재 기온은 {temp}도입니다."


agent = create_agent(
    model="openai:gpt-5.4-mini",
    tools=[get_weather],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "서울과 도쿄 기온 차이 알려줘"}]}
)

print(result["messages"][-1].content)