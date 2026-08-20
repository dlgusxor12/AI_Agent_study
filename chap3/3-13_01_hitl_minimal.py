from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    payload: str
    decision: str
    result: str


def prepare(state):
    return {"payload": "(중요) 본 결재 문서를 외부로 발송합니다."}


def approval(state):
    user_input = interrupt({
        "question": "다음 작업을 실행해도 될까요? (yes/no)",
        "payload": state["payload"],
    })
    return {"decision": str(user_input).strip().lower()}


def execute(state):
    if state["decision"] == "yes":
        return {"result": f"실행 완료: {state['payload']}"}
    return {"result": "사용자 거부로 작업이 취소되었습니다."}


builder = StateGraph(State)
builder.add_node("prepare", prepare)
builder.add_node("approval", approval)
builder.add_node("execute", execute)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "approval")
builder.add_edge("approval", "execute")
builder.add_edge("execute", END)

graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-1"}}

    first = graph.invoke(
        {"payload": "", "decision": "", "result": ""},
        config=config,
    )
    print("[1] interrupt에 노출된 값:")
    for item in first.get("__interrupt__", []):
        print("  ->", item.value)

    final = graph.invoke(Command(resume="yes"), config=config)
    print("[2] 최종 결과:", final["result"])