import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="acme_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 사내 규정 전문가야. 주어진 문서만 근거로 한국어로 간결히 답해. "
               "문서에 없는 내용이면 '문서에 없는 내용입니다'라고 답해."),
    ("human", "문서:\n{context}\n\n질문: {question}"),
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _format_docs(docs) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page", "?")
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{src}#page{page}] {d.page_content}")
    return "\n\n".join(parts)


@tool
def search_company_policy(question: str, k: int = 4) -> dict:
    """사내 규정 문서를 검색해 답변과 출처 목록을 함께 반환한다.

    회사 내부 규정(휴가, 근태, 보안 등) 관련 질문에 사용한다.
    검색 결과가 없으면 found=False로 반환된다.

    Args:
        question: 사내 규정에 대한 자연어 질문
        k: 검색할 문서 청크 수. 기본 4. 정확도가 부족하면 6~8로 늘려도 된다.

    Returns:
        {
            "found": bool,
            "answer": str,
            "sources": [{"source": str, "page": int}, ...]
        }
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    if not docs:
        return {
            "found": False,
            "answer": "관련 문서를 찾지 못했습니다.",
            "sources": [],
        }

    chain = prompt | llm
    msg = chain.invoke({"context": _format_docs(docs), "question": question})
    answer = msg.content if hasattr(msg, "content") else str(msg)

    sources = [
        {"source": d.metadata.get("source", "unknown"), "page": d.metadata.get("page")}
        for d in docs
    ]
    return {"found": True, "answer": answer, "sources": sources}


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_company_policy],
    prompt=(
        "너는 한국어로 답하는 회사 비서다. "
        "사내 규정 관련 질문에는 search_company_policy 도구를 사용한다. "
        "도구 결과의 sources 정보를 답변 마지막에 '출처: ...' 형식으로 짧게 인용한다."
    ),
)


def ask(q: str) -> None:
    print(f"\nQ: {q}")
    r = agent.invoke({"messages": [{"role": "user", "content": q}]})
    for m in r["messages"]:
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"  [도구 호출] {tc['name']}({tc['args']})")
    print(f"  A: {r['messages'][-1].content}")


ask("반차는 몇 시간인가요?")
ask("연차 휴가 규정을 자세히 알려줘. 가능하면 더 많은 문서를 참고해서.")
ask("우리 회사 화성 출장 규정은?")