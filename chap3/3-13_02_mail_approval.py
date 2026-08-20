import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()


class MailState(TypedDict):
    purpose: str
    draft: str
    decision: str
    sent: bool


llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
)


def draft_mail(state):
    prompt = (
        "다음 목적에 맞는 한국어 비즈니스 메일 초안을 작성하세요.\n"
        "- 제목과 본문을 함께\n"
        "- 본문은 5문장 이내\n"
        f"목적: {state['purpose']}"
    )
    return {"draft": llm.invoke(prompt).content}


def approve(state):
    decision = interrupt({
        "question": "이 초안을 발송해도 될까요? (ok / no)",
        "draft": state["draft"],
    })
    return {"decision": str(decision).strip().lower()}


def send_mail(state):
    if state["decision"] == "ok":
        # 실제 환경에서는 4-7의 SMTP 도구를 호출
        print("\n[메일 발송 시뮬레이션] -----------------------------")
        print(state["draft"])
        print("---------------------------------------------------")
        return {"sent": True}
    print("\n[발송 취소] 사용자가 승인하지 않았습니다.")
    return {"sent": False}


builder = StateGraph(MailState)
builder.add_node("draft_mail", draft_mail)
builder.add_node("approve", approve)
builder.add_node("send_mail", send_mail)
builder.add_edge(START, "draft_mail")
builder.add_edge("draft_mail", "approve")
builder.add_edge("approve", "send_mail")
builder.add_edge("send_mail", END)

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "mail-demo"}}

# 1) 초안 생성 → approve의 interrupt에서 정지
first = graph.invoke(
    {"purpose": "다음 주 화요일 14시 기획 미팅을 김부장님께 잡는 메일",
     "draft": "", "decision": "", "sent": False},
    config=config,
)
for item in first.get("__interrupt__", []):
    print(item.value["question"])
    print(item.value["draft"])

# 2) 사용자가 ok 라고 응답했다고 가정
final = graph.invoke(Command(resume="ok"), config=config)
print(f"sent={final['sent']}")