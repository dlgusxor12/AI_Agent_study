import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 숙련된 {role}이야. 답은 {max_lines}줄 이내로 간결하게."),
    ("user", "{question}"),
])

messages = prompt.format_messages(
    role="개발자 멘토",
    max_lines=3,
    question="Git rebase와 merge의 차이를 알려 줘.",
)

# 렌더링 결과 확인
for m in messages:
    print(f"[{m.type}] {m.content}")

print("\n=== LLM 답변 ===")
print(llm.invoke(messages).content)
