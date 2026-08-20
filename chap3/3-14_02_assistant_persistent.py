import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ─────────────────────────────────────────
# 1. 도구 3종 (3-10과 동일)
# ─────────────────────────────────────────
search_tool = DuckDuckGoSearchRun()


@tool
def calculator(expression: str) -> str:
    """사칙연산 수식을 평가합니다.

    Args:
        expression: 파이썬 문법의 산술식 (예: "(12+8)*3", "235*17")
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "허용되지 않은 문자가 포함되어 있습니다."
    try:
        value = eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        return f"계산 오류: {e}"
    return f"{expression} = {value}"


MEMO_DIR = Path("./notes")
MEMO_DIR.mkdir(exist_ok=True)
MEMO_FILE = MEMO_DIR / "memos.txt"


@tool
def save_memo(content: str) -> str:
    """사용자의 메모 한 줄을 텍스트 파일에 날짜와 함께 저장합니다.

    Args:
        content: 저장할 메모 내용 (예: "내일 10시 미팅")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    clean = re.sub(r"\s+", " ", content).strip()[:200]
    with MEMO_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {clean}\n")
    return f"메모 저장 완료: '{clean}'"


# ─────────────────────────────────────────
# 2. 영구 체크포인터 (← 3-10과 유일한 차이)
# ─────────────────────────────────────────
DB_PATH = "assistant.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
saver = SqliteSaver(conn)

SYSTEM_PROMPT = (
    "너는 한국어로 답하는 친절한 개인 비서다. "
    "다음 세 가지 도구 중 필요한 것을 골라 사용한다.\n"
    "- duckduckgo_search: 최신 정보·사실 확인이 필요할 때\n"
    "- calculator: 산수 계산이 필요할 때\n"
    "- save_memo: 사용자가 '메모', '저장', '기록' 등을 요청할 때\n"
    "도구 결과가 충분하면 바로 답하고, 출처가 있으면 간단히 명시한다."
)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_tool, calculator, save_memo],
    prompt=SYSTEM_PROMPT,
    checkpointer=saver,
)


def ask(user_text: str, thread_id: str = "user-1") -> str:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 15,
    }
    r = agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config=config,
    )
    return r["messages"][-1].content


if __name__ == "__main__":
    print("멀티툴 비서 (영구 저장 모드). 종료하려면 'quit'.\n")
    print(f"DB 파일: {DB_PATH}\n")
    while True:
        try:
            text = input("you > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "종료"):
            break
        try:
            print("bot >", ask(text), "\n")
        except Exception as e:
            print(f"[오류] {e}\n")