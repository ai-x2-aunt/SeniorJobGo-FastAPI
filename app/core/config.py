from dotenv import load_dotenv
import os
from pydantic_settings import BaseSettings
from typing import List

# 환경변수 로드
load_dotenv()

class Settings(BaseSettings):
    API_BASE_URL: str = "http://localhost:8000/api/v1"  # 백엔드 API 기본 URL
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    DATA_API_KEY: str = os.getenv("DATA_API_KEY")
    DATA_DECODING_KEY: str = os.getenv("DATA_DECODING_KEY")
    DATA_URL: str = os.getenv("DATA_URL")
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",  # React 기본 개발 서버
        "http://localhost:5173",  # Vite 개발 서버
        "http://127.0.0.1:5173"   # Vite 개발 서버 (IP 주소)

    ]
    # Tavily API 설정
    tavily_api_key: str

    class Config:
        env_file = ".env"
        extra = "allow"  # 추가 필드 허용

settings = Settings() 