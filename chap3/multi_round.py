import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()


@tool(parse_docstring=True)
def add(a: int, b: int) -> int:
    """두 정수를 더한 값을 반환합니다.

    Args:
        a: 첫 번째 정수
        b: 두 번째 정수
    """
    return a + b


@tool(parse_docstring=True)
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한 값을 반환합니다.

    Args:
        a: 첫 번째 정수
        b: 두 번째 정수
    """
    return a * b


agent = create_agent(model="openai:gpt-5.4-mini", tools=[add, multiply])

result = agent.invoke(
    {"messages": [{"role": "user", "content": "(3+4) 곱하기 5는?"}]}
)

for msg in result["messages"]:
    tag = type(msg).__name__
    tc = getattr(msg, "tool_calls", None)
    if tc:
        print(f"[{tag}] tool_calls={tc}")
    else:
        print(f"[{tag}] {msg.content}")