# 파일명: chat_stream_usage.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0.3,
    max_tokens=200,
    # 스트리밍에서도 사용량을 받으려면 이 옵션이 필요
    stream_usage=True,
)

messages = [
    SystemMessage(content="너는 경력 10년의 백엔드 개발자야."),
    HumanMessage(content="비동기(async)가 왜 필요한지 3줄로 설명해 줘."),
]

print("=== 스트리밍 출력 ===")
final_chunk = None
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
    final_chunk = chunk

print("\n\n=== 사용량 ===")
# 마지막 청크에 usage_metadata가 담겨 있음
if final_chunk and final_chunk.usage_metadata:
    meta = final_chunk.usage_metadata
    input_tokens = meta["input_tokens"]
    output_tokens = meta["output_tokens"]
    total = meta["total_tokens"]

    # 실제 비용을 추산할 때는 OpenAI 공식 가격 페이지에서 현재 단가를 가져와 채워 넣으세요.
    input_price_per_1m = 0.0
    output_price_per_1m = 0.0
    cost_usd = input_tokens * input_price_per_1m / 1_000_000 + output_tokens * output_price_per_1m / 1_000_000
    cost_krw = cost_usd * 1400  # 환율은 예시

    print(f"입력 토큰 : {input_tokens}")
    print(f"출력 토큰 : {output_tokens}")
    print(f"총 토큰   : {total}")
    print(f"예상 비용 : ${cost_usd:.6f} (약 {cost_krw:.2f}원)")
