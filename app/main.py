# 패키지 임포트
import json
import logging
import signal
import sys
import uvicorn
from contextlib import asynccontextmanager

# 패키지 임포트 (fastapi)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 모듈 임포트
from app.agents import initialize_agents
from app.core.config import settings
from app.core.prompts import EXTRACT_INFO_PROMPT
from app.services import initialize_vector_store
from app.routes import register_routes
from db import database_initialize, database_shutdown

# 로깅 설정을 더 자세하게
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 앱 초기화 및 종료 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        # 벡터 스토어 및 검색 객체 초기화 (.services/)
        logger.info("벡터 스토어(ingest) 및 검색 객체(search)를 초기화합니다.")
        initialize_vector_store(app)
        logger.info("벡터 스토어 및 검색 객체 초기화 완료")
        
        # LLM과 에이전트 초기화 (.agents/)
        logger.info("LLM과 에이전트를 초기화합니다.")
        initialize_agents(app)
        logger.info("LLM과 에이전트 초기화 완료")

        # 라우터 등록 (.routes/, ..db/)
        logger.info("데이터베이스 초기화 및 라우터를 등록합니다.")
        database_initialize(app)
        register_routes(app)
        logger.info("데이터베이스 초기화 및 라우터 등록 완료")

    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {str(e)}", exc_info=True)
        raise
        
    yield
    # 데이터베이스 종료
    logger.info("데이터베이스 종료 중...")
    database_shutdown()
    logger.info("데이터베이스 종료 완료")

    # shutdown
    logger.info("서버를 종료합니다...")

# FastAPI 앱 생성 시 lifespan 설정
app = FastAPI(lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def signal_handler(sig, frame):
    """
    시그널 핸들러 - SIGINT(Ctrl+C)나 SIGTERM 시그널을 받으면 실행됨
    sig: 발생한 시그널 번호
    frame: 현재 스택 프레임
    """
    logger.info(f"\n시그널 {sig} 감지. 서버를 안전하게 종료합니다...")
    sys.exit(0)

@app.post("/api/v1/extract_info/")
async def extract_user_info(request: dict):
    try:
        user_message = request.get("user_message", "")
        chat_history = request.get("chat_history", "")
        
        logger.info(f"[extract_info] 요청 메시지: {user_message}")
        logger.info(f"[extract_info] 대화 이력: {chat_history}")
        
        # LLM 호출
        response = app.state.llm.invoke(
            EXTRACT_INFO_PROMPT.format(
                user_query=user_message,
                chat_history=chat_history
            )
        )
        
        # 응답 처리
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"[extract_info] LLM 원본 응답: {response_text}")
        
        # JSON 파싱 전처리
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        logger.info(f"[extract_info] 정제된 응답: {cleaned}")
        
        # JSON 파싱
        try:
            info = json.loads(cleaned)
            logger.info(f"[extract_info] 파싱된 정보: {info}")
            return info
        except json.JSONDecodeError as e:
            logger.error(f"[extract_info] JSON 파싱 실패: {str(e)}")
            return {
                "직무": "",
                "지역": "",
                "연령대": ""
            }
            
    except Exception as e:
        logger.error(f"[extract_info] 처리 중 에러: {str(e)}", exc_info=True)
        return {
            "직무": "",
            "지역": "",
            "연령대": ""
        }

if __name__ == "__main__":
    # Ctrl+C와 SIGTERM 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 개발 서버 실행
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,  # 코드 변경 시 자동 재시작
            reload_delay=1
        )
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {str(e)}")
        sys.exit(1)