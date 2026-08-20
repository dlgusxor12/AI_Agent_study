> Part 3_1
### 에이전트의 3요소
- 에이전트 = LLM (판단 엔진) + Tools(행동 수단) + Loop(반복 제어)

### ReAct 패턴
- Reasoning(추론) 과 Acting(행동) 을 번갈아 시키는 프롬프트 전략
```
Thought: 지금 상황을 바라보고, 무엇을 해야 할지 추론한다
Action: 어떤 도구를 어떤 인자로 부를지 결정한다
Observation: 도구 실행 결과를 관찰한다
```

<br>

> Part 3_2
### Langgraph
- 상태 그래프(state graph) 기반 워크플로 엔진. 노드(node)가 상태(state)를 읽어 업데이트하고, 엣지(edge)를 따라 다음 노드로 이동
- `create_agent()` 이용해 그래프 구성

<br>

> Part 3_3
### Tool Calling
```
① 도구 등록
   - 파이썬에서 함수와 스키마(이름·설명·파라미터 타입)를 LLM에 전달

② 질문 입력
   - 사용자 메시지가 모델에 들어감

③ LLM의 응답(Action)
   - 모델이 "get_weather({city: 'Seoul'}) 불러줘"라는
     구조화된 JSON(= tool_calls)을 생성

④ 파이썬이 실제로 실행
   - tool_calls를 읽고 실제 파이썬 함수 호출

⑤ 결과를 다시 모델에게(Observation)
   - 함수 반환값을 ToolMessage로 감싸 모델에 넘김

⑥ 모델의 다음 판단
   - 답을 낼지, 도구를 또 부를지 결정
   - 필요하면 ②~⑤ 반복
```

### 도구 스키마
- 예시와 같은 json 메타데이터만 전송
```
{
  "name": "get_weather",
  "description": "도시 이름을 받아 현재 날씨(섭씨)를 반환합니다.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "도시 이름(예: \"Seoul\")"
      }
    },
    "required": ["city"]
  }
}
```

- parameter 구성

|스키마 필드|	어디에서 가져옴|
|---|---|
|`name`	|함수 이름|
|`description`	|함수 docstring|
|`parameters.type`	|타입 힌트 (str, int, list, ...)|
|`properties.*.description`	|docstring의 Args: 블록|
|`required` |	기본값이 없는 파라미터|

- 주의
  - docstring 입력 시 반드시 1줄 비우고 입력해야 함. 아니면 오류
  ```
  """도시 이름을 받아 현재 날씨(섭씨)를 반환합니다.

    Args:
        city: 도시 영문 이름 (예: "Seoul", "Tokyo")
  """
  ```

<br>

> Part 3_4
- 내장 도구
  - `langchain-community`에는 공식 커뮤니티 도구, `langchain-experimental`에는 보안상 주의가 필요한 실험적 도구 존재

> Part 3_6
- @tool 데코레이터
  - 예시

```
@tool(parse_docstring=True)
def convert_currency(amount: float, from_: str, to: str) -> float:
    """금액을 한 통화에서 다른 통화로 환전합니다.

    Args:
        amount: 환전할 금액 (예: 1000.0)
        from_: 원화 통화 코드 (예: "KRW", "USD")
        to: 목표 통화 코드 (예: "JPY", "EUR")
    """
    ...
```

<br>

> Part 3_8
### LangGraph의 InmemorySaver
- 체크포인터를 이용하여 이전 대화의 messages를 어딘가에 저장해두었다가 다음 호출에 이용
  - 예시

```
첫 호출
  ── 사용자: "내 이름은 홍길동"
  ── 에이전트: "안녕, 홍길동님"
  [체크포인터: {thread_id="user-1", messages=[...]}]

두 번째 호출 (같은 thread_id)
  [체크포인터에서 messages 로드]
  ── 사용자: "내 이름이 뭐였지?"
  ── 에이전트: "홍길동님이세요"
  [체크포인터 업데이트]
```

### `InMemorySaver` 위치

|체크포인터	|저장 위치	|프로세스 재시작|	대표 용도|
|---|---|---|---|
|InMemorySaver|	프로세스 메모리(딕셔너리)|	사라짐|	개발·테스트|
|SqliteSaver|	SQLite 파일	|유지|	개인/소규모 서비스|
|PostgresSaver|	Postgres DB	|유지|	프로덕션 |

- `thread_id`만 설정해주면 사용자 별 독립된 대화 설정 가능