import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

summarize_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "한 문장으로 요약해."),
        ("user", "{text}"),
    ]) | llm | parser
)

keywords_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "핵심 키워드 3개를 콤마로 구분해 출력. 다른 말 금지."),
        ("user", "{text}"),
    ]) | llm | parser
)

# 명시적 RunnableParallel 사용 (dict 단축 문법도 가능)
parallel = RunnableParallel(
    summary=summarize_chain,
    keywords=keywords_chain,
    original=RunnablePassthrough(),  # 원본까지 같이 보존
)

article = (
    "전기차 시장 점유율이 올해 처음으로 내연기관차를 역전했다. "
    "충전 인프라 확대와 보조금 정책이 주된 원인으로 꼽힌다."
)

result = parallel.invoke({"text": article})
for k, v in result.items():
    print(f"\n[{k}]")
    print(v)



