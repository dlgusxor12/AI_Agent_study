> Part 2_1

### RAG (Retrieval Augmented Generation)
- LLM이 답변을 생성하기 전 외부 지식 소스에서 관련 정보를 검색(Retrieval)하여, 해당 정보를 컨텍스트로 제공(Augmented)한 뒤 답변을 생성(Generation)하도록 하는 기법

- 필요 이유
  - Knowledge Cutoff : 학습 데이터 시점 이후의 정보를 LLM은 알지 못함
  - Hallucination : 모르는 것을 그럴싸하게 지어냄
  - 도메인/사내 지식 부재 : 학습에 포함되지 않은 특정 회사 문서, 최신 논문, 개인 데이터에 접근 불가
  - 재학습 비용 : 새 지식을 넣으려면 파인 튜닝 필요 → 비용, 효율 문제

### RAG 파이프라인
  - 1단계 : Indexing
    - 사전 준비 단계, 문서 준비 시 1회

    ```
    [원본 문서 준비] → [문서 로딩 (Load/Ingestion)] → (문서 Parsing / Extraction) → (Cleaning / Nomalization) → [청킹(Chunking)] → [임베딩(Embedding)] → [저장]
    ```

```
                    [원본 데이터]
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
       PDF              DOCX             Web
        │                │                │
        ↓                ↓                ↓
   PDF Parser       DOCX Parser      HTML Parser
        │                │                │
        └────────────────┼────────────────┘
                         ↓
                  [Text Extraction]
                         ↓
                  [Cleaning / OCR]
                         ↓
                    Document
                    ├─ content
                    └─ metadata
                         ↓
                     Chunking
                         ↓
                ┌────────┴────────┐
                ↓                 ↓
              Chunk            Metadata
                │                 │
                ↓                 │
            Embedding             │
                │                 │
                └────────┬────────┘
                         ↓
                    Vector DB
```


  - 2단계 : 검색(Retrieval) + 생성(Generation)
    - 사용자 질의가 들어온 시점에 실행

    ```
    [사용자 질문 입력] → [질의 임베딩] → [유사도 검색(Retrieval)] → [프롬프트 증강(Augmentation)] → [생성(Generation)]
    ```

<br>

> Part 2_2
## 1단계 - Document Load/Ingestion
### 정의
- 문서의 텍스트를 추출하는 단계
  - 2차원 이상의 의미 구조를 1차원 선형 시퀀스로 projection하는 손실 변환
  - 비정형·반정형 외부 지식원을 계산 가능한 표현으로 변환하는 전처리 단계
    - 산출물인 텍스트 + 보존된 구조 + 메타 데이터가 이후 과정의 품질을 결정

### 구조가 명시적인 포맷 (structured/semi-structured)
  - HTML, XML, Markdown, JSON 등 논리 구조가 명확한 포맷들은 구조 선택, 사영 문제
    1. 주어진 구조 중 무엇이 내용이고 무엇이 잡음인가 (selection)
        - 웹 페이지에서 사이드바의 광고, 저작권 표시 등 RAG 과정에 필요 없음
    2. 트리를 어떤 순서와 어떤 손실로 1차원 시퀀스로 펼칠 것인가 (linearization)
        - 트리에서 시퀀스로 가는 순간 존재했던 형제, 부모, 자식 등 관계가 소실
    3. 무엇을 하나의 문서 단위로 볼 것인가 (granularity)
  
### 구조가 암묵적인 포맷
  -  PDF같은 파일 : 구조가 명확하지 않음 (추출 후 문자, 좌표, 폰트 정보, 너비가 포함된 글리프 도출)
    - 문제 발생 예시
      1. 읽기 순서 붕괴
      2. 표(table) 파괴
      3. 스캔 문서와 OCR
      4. 헤더·푸터·각주·워터마크 오염
      5. 하이픈 분절

### `Document` 객체
- Loader(Langchain 내)는 `Document` 객체의 리스트를 반환
- Document 구성
  - 사용자 : `metadata`로 출처 등 확인  
  - LLM : `page_content`의 내용 활용
```
Document
├── page_content : str       ← 실제 검색·답변에 쓰일 텍스트
└── metadata     : dict      ← 출처, 페이지 번호, URL 등 출처 추적과 필터링에 쓰일 정보
```

- `metadata` 설계 (추천)

| 키	| 의미	| 사용처| 
| --- | --- | --- |
| `source`	| 원본 파일명 또는 URL	| 답변 하단 출처 표시 | 
| `page`	| PDF 페이지 번호	| "3페이지 참고" 표시 | 
| `doc_type`	| pdf, html, txt, md	| 문서 유형별 필터링 | 
| `title`	| 문서 제목	| 검색 결과를 사람이 이해하기 쉽게 표시 | 
| `created_at` 또는 `version`	| 문서 버전	| 최신 규정만 검색 |

> Part 2_3
## 1단계 - Chunking
### 정의
- 긴 문서를 검색 가능한 작은 조각인 chunk 단위로 쪼개는 것
- 글의 연결망을 강제로 절단하는 행위

### 역할
  1. 임베딩의 단위 : 벡터 하나로 변환되는 대상
  2. 검색의 단위 : 검색 결과로 뽑혀 나오는 최소 단위 (`chunk_size`)
  3. LLM이 읽는 단위 : 프롬프트에 들어가는 근거 

### 이유
  - 컨텍스트 윈도우 초과 : 한 청크가 한도를 넘으면 임베딩 자체가 실패
  -  검색 정밀도 저하 : 50페이지짜리 문서 전체를 하나의 벡터로 압축하면, 질문과 정말 관련 있는 2문단짜리 부분을 평균으로 희석시켜 버림

### 특징

|	| 작은 청크 |	큰 청크 |
| --- | --- | --- |
|임베딩 변별력|	좋음 (한 주제만 담김)	|나쁨 (의미 희석)|
|문맥 완결성|	나쁨 (지시어 끊김)|	좋음|
|근거 온전성|	나쁨 (답이 잘림)|	좋음|
|토큰 효율|	좋음|	나쁨|

- 정밀도 (precision) : 청크 중 실제로 유용한 부분의 비율 (검색에 필요) → 작은 청크가 유리 
- 재현율 (recall) : 답에 필요한 근거를 담은 비율 (생성에 필요) → 큰 청크가 유리

  → 검색할 때 쓰는 단위와 LLM에게 줄 단위를 다르게 하는 것이 중요

### 기본 전략

1. 고정 크기 분할
- 원리 : n개의 토큰 단위로 청킹 진행
- 문제 : 문장 중간이 끊김

2. overlap (sliding window)
- 원리 : 인접한 chunk를 일부 겹치게 함 (예) 512 토큰에 128 overlap -> 512 +- 128)
- 문제 : overlap 크기에 따라 인덱스의 크기가 변화

3. 재귀적 분할
- 원리 : 문단(\n) → 문장 → 단어 순으로 구분자에 우선순위를 두고 분할

4. 구조 기반 분할
- 원리 : Markdown 제목, HTML 태그, 코드의 함수 등을 경계를 두고 분할

5. 시멘틱 청킹
- 원리 : 문장들을 임베딩 → 인접 문장 간 유사도가 급락하는 지점을 경계로 삼고 분할 → PDF 등에서도 적용 가능

6. 추가 전략
- 문맥 주입 (Contextual Retrival) : 각 chunk 앞 LLM이 생성한 짧은 설명을 붙여 임베딩

`원본: "매출은 전분기 대비 3% 증가했다."`

`주입 후: "이 청크는 ACME사 2023년 2분기 실적 보고서의 매출 부문에서 발췌되었다. 직전 분기 매출은 3.14억 달러였다. 매출은 전분기 대비 3% 증가했다."`

- Late Chunking : 문서 전체를 인코더에 주입 → 모든 토큰이 문맥을 흡수한 표현을 얻음 → 청크 구간 별로 해당 구간을 평균

- 명제화 (Propositionization) : LLM으로 원문을 자족적인 원자 명제들로 재작성

원본 : `"아인슈타인은 1879년 울름에서 태어났다. 그는 1921년 노벨상을 받았다."`

주입 후 : `["아인슈타인은 1879년 울름에서 태어났다.", "아인슈타인은 1921년 노벨물리학상을 받았다."]`

### Langchain - Chunking
- `RecursiveCharacterTextSplitter` 이용

- 아이디어
  1. 큰 경계 분할 : `\n\n` (문단) → `\n` (문장) → `" "` (공백) → `""` (글자)
  2. 쪼갠 조각이 `chunk_size`를 넘으면 더 작은 경계로 재귀적으로 다시 분할
  3. 각 청크 사이에 `chunk_overlap`만큼 내용을 겹쳐 삽입

- 핵심 파라미터
  - `chunk_size` : 한 청크의 최대 글자 수
  - `chunk_overlap` : 인접 청크 간 겹치는 글자 수

<br>

> Part 2_4
## 1단계 - Embedding
### 정의
- 텍스트를 고정 크기의 실수 벡터로 변환하는 것
  - 의미가 비슷한 텍스트는 벡터 공간에서 가깝게, 다른 텍스트는 멀게 배치
  - 문서와 질문을 같은 임베딩 모델로 변환해야 좌표계가 일치

### 거리 계산
- 코사인 유사도 이용
  - (-1, 1) 범위, 1에 가까울 수록 강한 연관관계
  - 벡터를 단위 길이로 미리 L2 normalize 해두면 코사인 유사도 계산 시 내적만 진행하면 되서 계산 빨라짐
  $$cos(a,b)= \frac{a⋅b}{∥a∥∥b∥}$$

### Bi-encoder vs. Cross-encoder
- Bi-encoder (이중 인코더)
  - 질문과 문서를 각각 따로 인코딩해서 벡터를 만들고, 마지막에 코사인 유사도만 계산
  - 장점 : 문서 벡터를 미리 계산해서 저장 → 질의가 들어오면 질문만 인코딩하고 벡터 비교
  - 단점 : 질문가 문서가 서로를 보지 못한 채 벡터로 압축
```
질문 → [인코더] → q벡터  ┐
                        ├→ cos(q, d)
문서 → [인코더] → d벡터  ┘
```

- Cross-Encoder (교차 인코더)
  - 질문과 문서를 이어붙여서 한 번에 모델에 넣고, 관련성 점수 하나를 출력
  - 장점 : 질문의 모든 토큰이 문서의 모든 토큰과 어텐션으로 상호작용
  - 단점 : 미리 계산이 불가능. 질문이 들어와야 계산 시작 가능
```
[질문 + 문서] → [인코더] → 점수
```

- 개선 방안 (retrieve-then-rerank 설계)
  1. 1차 검색(Bi-encoder) :  100만 개 중 상위 50~100개를 빠르게 추림 (recall 중심)
  2. Reranking(Cross-encoder) : 그 50~100개만 정밀하게 재정렬 (precision 중심)

### `OpenAIEmbeddings`

|메서드	|입력|	반환|	용도|
|---|---|---|---|
|`embed_documents(texts)`|	`List[str]`|	`List[List[float]]`	|문서 인덱싱 시|
|`embed_query(text)`	|`str`|	`List[float]`|	질문 임베딩 시|

### `dimensions` (Matryoshka)
- n차원 모델에 `dimensions=` 설정에 따라 원하는 차원 부여 가능
  - 앞쪽 차원에 중요한 정보가 몰리도록 학습되어 있어서, 뒤를 잘라내도 성능 저하가 완만

> Part 2_5
## 벡터 저장소: Chroma
### 특징
1. Approximate Nearest Neighbor(ANN) Algorithm
  - 수십만 벡터에서도 상위 K를 ms 단위로 검색
2. 벡터와 원본 텍스트·메타데이터를 디스크에 영구 저장
  - 앱이 꺼져도 유지

### 저장 대상
|저장 대상|	설명|
|---|---|
|청크 텍스트|	LLM에게 근거로 넣을 원문 조각|
|임베딩 벡터|	검색에 사용할 숫자 좌표|
|메타데이터	|파일명, 페이지, 문서 유형, 버전|
|ID	|추가·삭제·업데이트를 위한 고유 식별자|

### Chroma 저장 구조
```
./chroma_db/
├── chroma.sqlite3         ← 메타데이터 인덱스
└── <collection-uuid>/     ← 실제 벡터 데이터
    ├── data_level0.bin
    ├── header.bin
    ├── length.bin
    └── link_lists.bin
```

- `persist_directory` : Chroma 인덱스를 디스크에 저장할 폴더
```
Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
```

<br>

> Part 2_6
## 2단계 - Retrieval
### `Retriever`
- 정의 : 질문을 받아 관련 `Document`목록을 돌려주는 인터페이스
```
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
docs = retriever.invoke("연차 신청 방법은?")
```

### 검색 방식
|방식	|동작	|장점|단점|
|---|---|---|---|
|`"similarity"` (기본)|	질문과 가장 유사한 K개를 그냥 돌려줌|	단순·빠름|	상위 K가 서로 거의 같은 내용이면 정보 낭비|
|`"mmr"` (Maximal Marginal Relevance)|	유사도 + 다양성 트레이드오프로 K개를 고름|	중복 제거·폭넓은 근거|	파라미터 튜닝 필요|

- 다양한 검색 방식

|질문 유형|	추천 방식|	이유|
|---|---|---|
|정확한 조항 하나를 찾는 질문|	similarity|	가장 가까운 청크를 빠르게 찾음|
|여러 관점이 필요한 질문|	MMR|	비슷한 청크만 반복되는 것을 줄임|
|사용자 질문이 짧거나 애매함|	Multi-Query	|질문을 여러 표현으로 바꿔 검색 범위 확장|
|검색 결과가 너무 긴 경우|	Compression|	가져온 문서에서 관련 부분만 압축|