# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# parser = StrOutputParser()

# summarize_prompt = ChatPromptTemplate.from_messages([
#     ("system", "문장을 한 줄로 요약해."),
#     ("user", "{text}"),
# ])
# summarize_chain = summarize_prompt | llm | parser

# # 입력을 그대로 유지하면서 summary 필드만 추가
# pipeline = RunnablePassthrough.assign(summary=summarize_chain)

# article = (
#     "지난주 공개된 새 모델은 같은 가격대 경쟁 모델 대비 한국어 번역 품질이 "
#     "크게 개선되었고, 토큰당 비용도 30% 낮아졌다."
# )

# output = pipeline.invoke({"text": article})
# print(output)

# 원본·요약·번역을 한꺼번에 보존
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

summary_prompt = ChatPromptTemplate.from_template(
    "다음 글을 3문장으로 요약해줘:\n{text}"
)
translate_prompt = ChatPromptTemplate.from_template(
    "다음 요약을 자연스러운 한국어로 번역해줘:\n{summary}"
)

summary_chain = summary_prompt | llm | parser
translate_chain = translate_prompt | llm | parser

pipeline = (
    RunnablePassthrough.assign(summary=summary_chain)
    | RunnablePassthrough.assign(
        korean=lambda x: translate_chain.invoke({"summary": x["summary"]})
    )
)

result = pipeline.invoke({"text": "LangChain lets developers compose LLM applications..."})
print(result)
