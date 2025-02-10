import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.routes import chat_router
from app.agents.job_advisor import JobAdvisorAgent
from app.services.vector_store_ingest import VectorStoreIngest
from app.services.vector_store_search import VectorStoreSearch
from app.utils.embeddings import SentenceTransformerEmbeddings

import signal
import sys
import json
import logging
from contextlib import asynccontextmanager
from app.core.prompts import EXTRACT_INFO_PROMPT

# 로깅 설정을 더 자세하게
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        # 임베딩 모델 초기화 - 공통으로 사용
        embedding_model = SentenceTransformerEmbeddings("nlpai-lab/KURE-v1")
        
        logger.info("벡터 스토어를 초기화합니다. (ingest)")
        ingest = VectorStoreIngest(embedding_model=embedding_model)
        collection = ingest.setup_vector_store()
        
        logger.info("벡터 스토어 검색 객체를 초기화합니다. (search)")
        vector_search = VectorStoreSearch(
            collection=collection,
            embedding_model=embedding_model
        )
        
        logger.info("LLM과 에이전트를 초기화합니다.")
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7,
            request_timeout=30
        )
        
        app.state.job_advisor_agent = JobAdvisorAgent(
            llm=llm,
            vector_search=vector_search  # 검색 전용 객체 주입
        )
        logger.info("초기화 완료")
        
        
    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {str(e)}", exc_info=True)
        raise
        
    yield
    
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

# 라우터 등록
app.include_router(chat_router.router, prefix="/api/v1")

@app.post("/api/v1/extract_info/")
async def extract_user_info(request: dict):
    try:
        user_message = request.get("user_message", "")
        response = app.state.job_advisor_agent.llm.invoke(
            EXTRACT_INFO_PROMPT.format(query=user_message)
        )
        info = json.loads(response)
        return info
    except Exception as e:
        logger.error(f"Info extraction error: {e}")
        return {}

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