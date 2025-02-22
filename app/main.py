import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.training_advisor import TrainingAdvisorAgent
from app.services.vector_store_ingest import VectorStoreIngest
from app.services.vector_store_search import VectorStoreSearch
import signal
import sys
import json
import logging
from contextlib import asynccontextmanager
from app.core.prompts import EXTRACT_INFO_PROMPT
from db import database_initialize, database_shutdown
from app.routes import userInform_router
from app.routes import training_router
from app.agents.supervisor_agent import SupervisorAgent
from app.agents.chat_agent import ChatAgent
from app.services.meal_data_client import PublicDataClient

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
        # 벡터 스토어 초기화
        logger.info("벡터 스토어를 초기화합니다. (ingest)")
        ingest = VectorStoreIngest()  # DB 생성/로드 담당
        collection = ingest.setup_vector_store()  # Chroma 객체
        
        logger.info("벡터 스토어 검색 객체를 초기화합니다. (search)")
        vector_search = VectorStoreSearch(collection)
        app.state.vector_search = vector_search  # 앱 상태에 저장

        logger.info("벡터 스토어 및 검색 객체 초기화 완료")

        
        # LLM과 에이전트 초기화
        logger.info("LLM과 에이전트를 초기화합니다.")
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.5,
            request_timeout=30
        )
        app.state.llm = llm  # 앱 상태에 저장
        
        # 에이전트 초기화
        app.state.supervisor = SupervisorAgent(llm)
        app.state.job_advisor = JobAdvisorAgent(llm, vector_search)
        app.state.training_advisor = TrainingAdvisorAgent(llm)
        app.state.chat_agent = ChatAgent(llm)  # ChatAgent 추가

        logger.info("LLM과 에이전트 초기화 완료")

        # 라우터 등록
        logger.info("데이터베이스 초기화 및 라우터를 등록합니다.")
        database_initialize(app)
        from app.routes import chat_router
        app.include_router(chat_router.router, prefix="/api/v1")
        logger.info("데이터베이스 초기화 및 라우터 등록 완료")

        # data_client 초기화 및 등록
        app.state.data_client = PublicDataClient()

    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {str(e)}", exc_info=True)
        raise
        
    yield
    # 데이터베이스 종료
    database_shutdown()

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
app.include_router(userInform_router.router)
app.include_router(training_router.router)

def signal_handler(sig, frame):
    """
    시그널 핸들러 - SIGINT(Ctrl+C)나 SIGTERM 시그널을 받으면 실행됨
    sig: 발생한 시그널 번호
    frame: 현재 스택 프레임
    """
    logger.info(f"\n시그널 {sig} 감지. 서버를 안전하게 종료합니다...")
    sys.exit(0)


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