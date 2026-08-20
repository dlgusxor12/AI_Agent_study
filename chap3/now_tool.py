import os
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()


@tool(parse_docstring=True)
def get_current_time(timezone: str = "Asia/Seoul") -> str:
    """현재 시각을 지정된 시간대로 반환합니다.

    Args:
        timezone: IANA 시간대 이름 (예: "Asia/Seoul", "UTC", "America/New_York")
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        return f"알 수 없는 시간대: {timezone}"
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


agent = create_agent(model="openai:gpt-5.4-mini", tools=[get_current_time])

result = agent.invoke({
    "messages": [{"role": "user", "content": "지금 서울 몇 시야?"}]
})
print(result["messages"][-1].content)