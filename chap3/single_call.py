import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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


llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
llm_with_tools = llm.bind_tools([add])

msg = llm_with_tools.invoke("17 더하기 25는?")
print("content:", repr(msg.content))
print("tool_calls:", msg.tool_calls)