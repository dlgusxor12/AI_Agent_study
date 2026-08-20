import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

agent = create_agent(
    model="openai:gpt-5.4-mini",
    tools=[DuckDuckGoSearchRun()],
    checkpointer=InMemorySaver(),
    system_prompt="너는 한국어로 답하는 리서치 비서다.",
)

config = {"configurable": {"thread_id": "reader-1"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "내 관심사는 와인이야."}]},
    config=config,
)
r = agent.invoke(
    {"messages": [{"role": "user", "content": "최근 볼 만한 소식 하나 찾아줘."}]},
    config=config,
)
print(r["messages"][-1].content)