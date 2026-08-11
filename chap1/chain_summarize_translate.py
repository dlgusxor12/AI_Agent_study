import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# ---------- 1단계: 영문 요약 ----------
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following text in exactly 3 English sentences."),
    ("user", "{text}"),
])
summarize_chain = summarize_prompt | llm | parser

# ---------- 2단계: 한국어 번역 ----------
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 영어 문장을 자연스러운 한국어로 번역해. 번역문만 출력."),
    ("user", "{summary}"),
])
translate_chain = translate_prompt | llm | parser

# ---------- 합성: 입력 → 요약 → 번역 ----------
# {"text": "..."} 를 받아서
# summarize_chain으로 summary 생성
# 그 결과를 translate_chain의 "summary" 인자로 전달
full_chain = (
    {"summary": summarize_chain}
    | translate_chain
)

# 테스트 입력
article = (
    "LangChain is an open-source framework that helps developers build "
    "applications powered by large language models. It provides abstractions "
    "for prompts, chains, retrievers, tools, and agents. With the rise of "
    "LangGraph, developers can now build stateful, multi-actor applications "
    "using graph-based orchestration. The ecosystem continues to evolve "
    "rapidly, with new integrations and tools added regularly."
)

result = full_chain.invoke({"text": article})
print(result)