# Part 0

## AI Agent
- LLM이 스스로 도구를 골라 여러번 호출
- LLM + Tools + Loop

### Tool
- LLM이 호출할 수 있는 함수들
- 웹검색, 계산기, DB 조회, 파일 시스템, API 호출 등
- LLM은 도구 자체를 실행하지 않고 외부 시스템(LangChain, Langgraph 등)이 실제 함수를 호출

### Loop
- 답이 나올 때까지 도는 제어 흐름

## ReAct 패턴
- Reasoning (추론) + Acting (행동)
- ReAct: Synergizing Reasoning and Acting in Language Models 논문에서 처음 제시

### ReAct의 4단계
- Thought (사고) : LLM이 현재 상황을 정리하고 다음에 할 일을 머릿속으로 계획
- Action (행동) : 어떤 도구를 어떤 인자로 호출할지 결정해 신호
- Observation (관찰) : 도구가 돌려준 결과를 받아 들이고 정보로 흡수
- Answer (답) : 충분한 정보가 모였다고 판단되면 사용자에게 최종 답을 반환