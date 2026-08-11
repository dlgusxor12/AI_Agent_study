import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
parser = StrOutputParser()

kor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "정중한 한국어 존댓말로 한 문장으로 답해."),
        ("user", "{q}"),
    ]) | llm | parser
)

eng_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Answer in one friendly English sentence."),
        ("user", "{q}"),
    ]) | llm | parser
)

def detect_and_route(input_dict: dict):
    """한글 비율로 간단 감지 후 체인 선택."""
    q = input_dict["q"]
    hangul = sum(1 for c in q if "가" <= c <= "힣")
    chosen = kor_chain if hangul > len(q) * 0.3 else eng_chain
    return chosen.invoke(input_dict)

router = RunnableLambda(detect_and_route)

for question in ["LangChain이 뭐야?", "What is LangGraph?"]:
    print(f"\nQ: {question}")
    print(f"A: {router.invoke({'q': question})}")