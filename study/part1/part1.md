## 핵심 개념
> Part 1_1
### LM (Language Model) 
- 단어 시퀀스가 주어졌을 때 다음 단어의 확률 분포를 계산하는 시스템

- 다음 토큰 확률 예측 
$$ P(w_t​∣w_1​,w_2​,…,w_{t−1}​) $$

- 전체 문장의 확률 예측 
$$ P(w_1​,…,w_n​)=∏^{n}_{t=1}​P(w_t​∣w_{<t​}) $$

### LLM (Large Language Model)
- LM을 수십억 ~ 수조 개의 parameter, 방대한 text corpus, transformer architecture로 스케일업 한 것

### Token
- LLM이 텍스트를 처리하는 최소 단위
- 모델은 글자나 단어를 직접 다루지 않고, 텍스트를 토큰 시퀀스로 쪼갠 뒤 각 토큰을 정수 ID로 변환하여 처리
  - 영어 : 1 token → 4글자 → 약 0.75 단어
  - 한국어 : 1글자 → 1~3 token

### Context Window
- 모델이 한 번의 추론에서 볼 수 있는 최대 토큰 수
- 입력과 출력을 모두 합친 총량이 context window 한계 안에 들어와야 함
  - 한계를 넘으면 가장 오래된 부분이 잘려나가거나, 요청 거부
> [시스템 프롬프트] + [대화 기록] + [사용자 질문] + [모델 응답]
- 이유 : Self-Attention의 제곱 비용
  - Self-Attention의 계산 복잡도
    $$연산량∝O(n^2⋅d)$$
    $$(n: 시퀀스 길이, d: 임베딩 차원)$$
  - 각 토큰이 다른 모든 토큰과의 관련성을 계산하기 때문에, 토큰과 메모리 비용이 지수적 관계
    ex) 1,000 토큰 → 100만 번의 관계 계산
- 긴 입력에 따른 비용
  1. 금전 비용 : 입력 토큰이 길수록 비용이 증가
  2. 지연 시간 : 모델이 읽어야 할 내용이 많아져 응답 속도 지연
  3. 주의 분산 : 관련 없는 문서가 섞이면 중요한 근거를 놓칠 수 있음

### temperature
- 확률 분포의 분포를 조절
  - 낮은 temperature → 분포가 뾰족해짐 → 고확률 토큰에 집중 → 결정적·보수적·일관적
  - 높은 temperature → 분포가 평평해짐 → 저확률 토큰도 뽑힐 기회 증가 → 무작위·창의적·다양성
- 원리 : Softmax 식에 개입
  - Softmax
  $$P_i​=\frac{exp(z_i​/T)}{∑_j​exp(z_j​/T)}$$
  - T<1: logit을 크게 만듦 → 큰 값과 작은 값의 차이가 벌어짐 → 분포 뾰족
  - T=1: 원본 분포 그대로 (모델이 학습한 확률 그대로)
  - T>1: logit을 작게 만듦 → 값들의 차이가 좁혀짐 → 분포 평평
  - T→0: 최고 확률 토큰만 남음 → Greedy 디코딩과 동일 (완전 결정적)

- 예시
<div align="center">

| Temperature |	토큰 A | 토큰 B | 토큰 C | 특성 |
| --- | --- | --- | --- | --- |
| T = 0.5 | 78% |	11% |	6% | A에 강하게 집중 |
| T = 1.0 | 59% |	22% |	13% |	원본 분포 |
| T = 2.0 | 44% |	27% |	21% |	고르게 퍼짐 |

</div>

### Reasoning Effort
- 추론 (Reasoning) 모델이 답을 내기 전 내부적으로 "생각"에 얼마나 많은 연산을 쓸지 조절
- 모델이 답을 내기 전 사고 과정의 깊이·길이를 조절
- 회사마다 명칭 다를 수 있음 (openai : reasoning effort, claude : effort, ...)

<br>

> Part 1_2

### ChatOpenAI
- 예시 코드
```
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = "gpt-4o-mini"


def make_llm(temperature: float = 0, max_tokens: int = 500) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME, # 모델명
        temperature=temperature, # 무작위성
        max_tokens=max_tokens, # 최대 토큰 설정
        api_key=os.getenv("OPENAI_API_KEY"), # API 키 마운트
        # n = 3, 후보 답변 3개 출력 가능
    )


llm = make_llm()
response = llm.invoke("LangChain을 한 문장으로 설명해줘.")
print(response.content)
print(response.usage_metadata)
```
- ```api_key=```를 명시하지 않으면 ChatOpenAI()가 자동으로 환경변수 OPENAI_API_KEY를 탐색
  <br>  → load_dotenv()만 호출해도 동작


### 메시지 종류
| 역할 | 클래스 | 용도 | 넣을 내용 | 피해야 할 내용 |
| --- | --- | --- | --- | --- |
|system |	SystemMessage	| 모델의 정체성·규칙·제약 사항 지시	| 변하지 않는 규칙, 톤, 금지사항, 출력 형식 |	매번 바뀌는 사용자 질문 |
|user	| HumanMessage |	사용자의 질문·요청 | 이번 요청, 입력 데이터, 질문 | 장기 정책 |
|assistant |	AIMessage | 이전 대화에서 모델이 했던 답(멀티턴에 사용) | 이전 모델 답변, 대화 히스토리 |	사람이 만든 확정 규칙 |

- 무조건 지켜야하는 규칙 : SystemMessage
- 이번에 처리할 데이터 : HumanMessage

### invoke, stream
- invoke(input) : 답이 전부 생성될 때 까지 대기 후 한번에 반환
- stream(input) : 토큰이 생성되는대로 조금씩 흘려보냄

<br>

> Part 1_3
### LangChain 구성요소
- LangChain → 파이프라인 도구 모음
- 주요 구성 요소

| 구성 요소 | 역할 | 대표 클래스·함수 |
| --- | --- | --- |
| Models | LLM(또는 임베딩 모델) 호출 창구 | `ChatOpenAI`, `OpenAIEmbeddings` |
| Prompts | 템플릿에 변수를 끼워 넣어 프롬프트 문자열(또는 메시지)을 만듦 | `ChatPromptTemplate`, `PromptTemplate` |
| Output Parsers | 모델이 돌려준 텍스트를 파이썬 객체(JSON, Pydantic 등)로 변환 | `StrOutputParser`, `PydanticOutputParser`, `with_structured_output` |
| Tools | LLM이 호출할 수 있는 "함수". 외부 API·계산·DB 접근 | `@tool`, `create_agent`의 `tools` |
| Retrievers | 질문과 의미가 비슷한 문서를 벡터 저장소에서 검색 | `VectorStore.as_retriever()` |
| Chains (LCEL) | 파이프 연산자(\|)로 위 부품들을 연결해 실행 흐름을 정의 | `RunnableSequence`, `RunnableParallel` |
| Agents | LLM이 스스로 어떤 Tool을 호출할지 고르며 반복하는 구조 | `create_agent` (LangGraph) |
| Memory | 이전 대화·상태를 유지. 스레드 단위 체크포인트 | `MemorySaver`, `InMemoryChatMessageHistory` |

- 대략적인 파이프라인

```
사용자 입력
  ↓
Prompt: 입력을 모델이 이해할 메시지로 변환
  ↓
Model: 메시지를 읽고 응답 생성
  ↓
Parser: 자유 텍스트를 문자열, JSON, Pydantic 객체로 변환
  ↓
Application: 화면 출력, DB 저장, 다음 체인 입력
```

<br>

> Part 1_4

### 프롬프트 템플릿
- 같은 질문 구조를 여러 번 사용할 때 하드코딩 방식이 아닌 변수화해 사용하는 것
- 적용 x
```
llm.invoke("다음 리뷰를 긍정/부정으로 분류해 주세요.\n리뷰: 배송이 늦었지만 품질은 만족합니다")
llm.invoke("다음 리뷰를 긍정/부정으로 분류해 주세요.\n리뷰: 최악이에요 환불 요청합니다")
```
- 적용 o : {review} 자리에만 문자열을 교체하여 사용
```
prompt = ChatPromptTemplate.from_messages([
    ("system", "리뷰를 긍정/부정으로 분류해 주세요."),
    ("user", "리뷰: {review}"),
])
```

### ChatPromptTemplate vs PromptTemplate
| 클래스 | 출력 형태 |	사용 |
| --- | --- | --- |
| ChatPromptTemplate |	메시지 리스트 (system/user/assistant) |	챗 모델(ChatOpenAI 등)에 거의 항상 사용. 기본 선택지 |
| PromptTemplate |	단일 문자열 |	레거시 LLM, 또는 문자열 하나만 필요한 경우 |

- 보통의 경우 ChatPromptTemplate에서 system message에 변하지 않는 규칙, 사용자 입력에는 텍스트를 입력
- 예시
```
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 보안 원칙을 지키는 업무 메일 작성 도우미다. 개인정보는 마스킹한다."),
    ("human", "다음 메일에 대한 답변 초안을 작성해줘:\n{email}"),
])
```

### 메시지 역할
|역할 문자열 |	의미 |
| --- | --- |
|`"system"` |	모델의 정체성·규칙 (페르소나, 출력 형식 지시 등) |
|`"user"` / `"human"` |	사용자가 보낸 메시지 |
|`"assistant"` / `"ai"` |	이전 대화에서 모델이 했던 답 (멀티턴 예시 제공 시 사용) |

### 사용 함수
- `format_messages()`
  - 템플릿을 실행 가능한 메시지 리스트로 렌더링 하는 함수
```
msgs = prompt.format_messages(role="고양이 박사", topic="발바닥", style="귀여운")
# msgs → [SystemMessage(...), HumanMessage(...)]
llm.invoke(msgs)
```

- `partial()`
  - 일부 변수를 미리 고정하는 함수
```
base = ChatPromptTemplate.from_messages([
    ("system", "너는 {role}이야. 답변은 {language}로 해줘."),
    ("user", "{question}"),
])

# 언어를 한국어로 고정한 새 템플릿
korean_prompt = base.partial(language="한국어")

# 이제 role과 question만 채우면 됨
msgs = korean_prompt.format_messages(role="요리사", question="계란말이 꿀팁 있나요?")
```

<br>

> Part 1_5

### 프롬프트 엔지니어링 기법
### zero-shot
  - 예시 없이 지시만 주고 바로 시키는 방식
```
다음 문장의 감정을 긍정/부정으로 분류하세요.

"이 영화 정말 최고였어!"
```
  - 원리 : 모델이 pre-training 과정에서 이미 해당 과제 유형을 접했을 것이라 가정하고, 예시 없이 지시만으로 수행하게 함
  - 장점 : 프롬프트가 짧고 간단, 토큰 절약
  - 단점 : 과제가 모호하거나 특수한 출력 형식이 필요하면 성능이 불안정

### few-shot
- 몇 개의 예시를 프롬프트에 넣어 패턴을 보여주는 방식
- 예시가 1개면 one-shot, 여러 개면 few-shot
- 예시는 3개 정도가 효과적 (너무 많으면 편향 가능성)
```
문장의 감정을 분류하세요.

문장: "서비스가 너무 느렸다." → 부정
문장: "배송이 빠르고 만족스러워요." → 긍정
문장: "그냥 그랬어요." → 중립
문장: "다시는 안 살 거예요." →
```
  - 원리 : In-Context Learning 기법. 내부 parameter를 변경하지 않고 프롬프트 내 예시만으로 과제를 학습
  - 장점 : 출력 형식·톤·기준을 예시로 정확히 통제 가능
  - 단점 : 토큰 소모 증가, 예시 선택·순서에 성능이 민감

### CoT (Chain of Thought)
- 모델이 답을 바로 내지 않고 중간 추론 단계를 단계별로 서술하게 유도하는 기법
  - 추론 모델은 CoT를 프롬프트로 유도하지 않아도 모델 내부에서 자동으로 수행하도록 학습
```
Q: 카페에 사과가 23개 있었다. 20개를 팔고 6개를 더 들여왔다.
   지금 사과는 몇 개인가? 단계별로 생각해봐.

A: 처음에 23개 있었다.
   20개를 팔았으니 23 - 20 = 3개.
   6개를 더 들여왔으니 3 + 6 = 9개.
   따라서 답은 9개다.
```
  - zero-shot CoT : "Let's think step by step"같은 문장으로 추론을 유도하는 방식
  - few-shot CoT : 추론 과정이 포함된 예시를 몇 개 보여주고, 해당 스타일을 따라하도록 유도하는 방식
  
  - 확장
    - Self-Consistency : CoT를 여러 번 수행해 나온 답들 중 다수결로 최종 답을 선택하는 방식
    - ToT(Tree of Thought) : 추론을 한 줄기가 아니라 여러 갈래로 탐색하는 방식

### Role Prompting
- 모델에게 특정 역할·페르소나를 부여해 해당 관점·톤·전문성으로 답하게 하는 기법
```
당신은 20년 경력의 시니어 백엔드 엔지니어입니다.
주니어 개발자가 이해할 수 있도록, 아래 코드의 문제점을
친절하지만 정확하게 리뷰해 주세요.

[코드]
```
  - 원리 : 열할을 지정하면 모델의 응답이 그 역할에 맞는 스타일로 유도됨. pre-traning 과정에서 학습한 분포를 활성화
  - system 메시지로 role 부여하는 방법이 효과적

### Good Prompt
- 기법들을 다양하게 섞어서 사용하는 것도 효과적
```
[Role] 당신은 수학 교사입니다.
[Few-shot + CoT] 아래 예시처럼 단계별로 풀어주세요.
  예시: (추론 과정 포함된 예시)...
[질문] 이제 다음 문제를 풀어주세요: ...
```

- 5요소

| 요소 |	질문 |
| --- | --- |
| 역할 |	모델이 어떤 관점으로 답해야 하는가 |
| 목표	| 최종 산출물이 무엇인가 |
| 입력	| 어떤 데이터를 근거로 삼아야 하는가 |
| 제약	| 길이, 톤, 금지사항, 형식은 무엇인가 |
| 평가 기준	| 좋은 답과 나쁜 답을 어떻게 구분하는가 |

- 프롬프트 개선 방법
  1. 명확한 지시
  2. 예시 제공
  3. 출력 형식 지정
