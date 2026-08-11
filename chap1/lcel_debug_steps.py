# 파일명: lcel_debug_steps.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "문장을 한 단어로 감정 분류해. 긍정/부정/중립 중 하나."),
    ("user", "{text}"),
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

inputs = {"text": "상품은 좋은데 배송이 너무 늦었어요."}

# 1) 프롬프트 렌더링 결과
rendered = prompt.invoke(inputs)
print("=== [1] 프롬프트 결과 ===")
for m in rendered.to_messages():
    print(f"[{m.type}] {m.content}")

# 2) LLM 원본 응답
ai_msg = llm.invoke(rendered)
print("\n=== [2] LLM 원본 응답 (AIMessage) ===")
print(ai_msg)

# 3) 파서 통과 후 문자열
final = parser.invoke(ai_msg)
print("\n=== [3] 파서 통과 후 ===")
print(repr(final))

# 4) 전체 체인으로 한 번에
print("\n=== [4] 전체 체인 ===")
chain = prompt | llm | parser
print(repr(chain.invoke(inputs)))
