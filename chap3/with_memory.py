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

config = {"configurable": {"thread_id": "user-1"}}

r1 = agent.invoke(
    {"messages": [{"role": "user", "content": "내 이름은 홍길동이야."}]},
    config=config,
)
print("A1:", r1["messages"][-1].content)

r2 = agent.invoke(
    {"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]},
    config=config,
)
print("A2:", r2["messages"][-1].content)