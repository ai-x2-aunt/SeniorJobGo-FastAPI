# SeniorJobGo
---
- 고령층 맞춤 일자리부터 복지 정보까지, 시니어를 위한 AI 챗봇 서비스입니다.
![시니어잡고 이미지](./images/Senior_JobGo.png)

## 프로젝트 진행 기간📅
---
- 2025.01.08 ~ 2025.02.21

## Backend(FastAPI)
```plaintext
FastApi_SeniorJobGo/
├── app/                           # 백엔드 애플리케이션 핵심 로직
│   ├── __init__.py                # 패키지 초기화 파일
│   ├── main.py                    # FastAPI 앱 설정 및 실행 파일
│   ├── routes/                    # API 엔드포인트 정의
│   │   ├── __init__.py            # 패키지 초기화 파일
│   │   └── chat_router.py         # 채팅 관련 엔드포인트 (/api/v1/chat/)
│   ├── services/                  # 비즈니스 로직 및 서비스 레이어
│   │   ├── __init__.py            # 패키지 초기화 파일
│   │   ├── vector_store_ingest.py # 벡터 스토어 설정 및 Vector DB 저장
|   |   ├── vector_store_search.py # 벡터 Search 작업
│   │   └── conversation.py        # 대화 기록 관리 및 컨텍스트 처리
│   ├── agents/                    # LangGraph 에이전트 및 워크플로우
│   │   ├── __init__.py            # 패키지 초기화 파일
│   │   └── job_advisor.py         # LangGraph 기반 채용 조언 에이전트
│   ├── models/                    # 데이터 모델 및 스키마 정의
│   │   ├── __init__.py            # 패키지 초기화 파일
│   │   └── schemas.py             # Pydantic 모델 (ChatRequest, ChatResponse 등)
│   ├── core/                      # 애플리케이션 설정 및 공통 기능
│   │   ├── __init__.py            # 패키지 초기화 파일
│   │   ├── config.py              # 환경 변수 및 설정 관리
│   │   └── prompts.py             # LLM 프롬프트 템플릿 모음
│   └── utils/                     # 유틸리티 함수 및 상수
│       ├── __init__.py            # 패키지 초기화 파일
│       └── constants.py           # 상수 정의 (예: 딕셔너리, 고정값)
├── documents/                     # 초기 데이터 파일 저장소
│   └── jobs.json                  # 채용 정보 JSON 파일
├── jobs_collection/               # 벡터 DB 저장소 (FAISS 인덱스 등)
├── requirements.txt               # 프로젝트 의존성 패키지 목록
├── .env                           # 환경 변수 파일
└── README.md                      # 프로젝트 설명 문서
```

### 디렉토리 설명
---
- `app/`: 메인 애플리케이션 코드
  - `routes/`: API 엔드포인트 정의
  - `services/`: 비즈니스 로직 구현
  - `agents/`: LangGraph 에이전트 구현
  - `models/`: Pydantic 모델 정의
  - `core/`: 핵심 설정 및 프롬프트
  - `utils/`: 유틸리티 함수 및 상수

- `documents/`: 데이터 파일 저장소
- `jobs_collection/`: 벡터 데이터베이스 저장소


### 백엔드 설치 및 실행
---
1. 환경 설정
`bash
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt`

2. 환경 변수 설정
`bash
cp .env.example .env`

3. 서버 실행
`bash
uvicorn app.main:app --reload`

### API 문서
---
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
