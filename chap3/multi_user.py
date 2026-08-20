import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

agent = create_agent(
    model="openai:gpt-5.4-mini",
    tools=[],
    checkpointer=InMemorySaver(),
)

def chat(user_id: str, text: str) -> str:
    config = {"configurable": {"thread_id": user_id}}
    r = agent.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
    return r["messages"][-1].content


print("A:", chat("alice", "안녕, 내 이름은 앨리스야."))
print("B:", chat("bob",   "안녕, 내 이름은 밥이야."))
print("A:", chat("alice", "내 이름 뭐였지?"))
print("B:", chat("bob",   "내 이름 뭐였지?"))