from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv()

class Settings:
    API_BASE_URL: str = "http://localhost:8000/api/v1"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    DATA_API_KEY: str = os.getenv("DATA_API_KEY")
    DATA_DECODING_KEY: str = os.getenv("DATA_DECODING_KEY")
    DATA_URL: str = os.getenv("DATA_URL")
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",  # React 기본 개발 서버
        "http://localhost:5173",  # Vite 개발 서버
        "http://127.0.0.1:5173"   # Vite 개발 서버 (IP 주소)
    ]

settings = Settings() 