import os
import sqlite3
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

DB_PATH = "agent.db"
THREAD_ID = "demo-thread-1"

# 명시적 connection (프로세스 종료 전까지 살아 있어야 함)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
saver = SqliteSaver(conn)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=saver,
)

config = {"configurable": {"thread_id": THREAD_ID}}


def ask(text: str) -> str:
    r = agent.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
    return r["messages"][-1].content


if __name__ == "__main__":
    # 첫 실행이라면 sys.argv[1]이 'first', 두 번째면 'second'
    mode = sys.argv[1] if len(sys.argv) > 1 else "first"

    if mode == "first":
        print("[1차] 사용자: 내 이름은 홍길동, 취미는 등산이야.")
        print("bot >", ask("내 이름은 홍길동, 취미는 등산이야."))
        print("\n프로세스를 종료한 뒤, 같은 명령에 'second' 인자로 다시 실행하세요.")
    elif mode == "second":
        print("[2차 — 재시작 후] 사용자: 내 이름과 취미가 뭐였지?")
        print("bot >", ask("내 이름과 취미가 뭐였지?"))
    else:
        print("usage: python 3-14_01_sqlite_saver_basic.py [first|second]")