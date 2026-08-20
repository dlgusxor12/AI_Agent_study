import os
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

agent = create_agent(model="openai:gpt-5.4-mini", tools=[])

r1 = agent.invoke({"messages": [{"role": "user", "content": "내 이름은 홍길동이야."}]})
print("A1:", r1["messages"][-1].content)

r2 = agent.invoke({"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]})
print("A2:", r2["messages"][-1].content)