from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # 환경 설정
    ENV: str = os.getenv("ENV", "development")  # 기본값은 development
    
    # MongoDB 설정
    DEV_MONGO_URI: str = os.getenv("DEV_MONGO_URI", "mongodb://localhost:27017")
    DEV_DATABASE_NAME: str = os.getenv("DEV_DATABASE_NAME", "seniorjobgo")
    PROD_MONGO_URI: str = os.getenv("PROD_MONGO_URI")
    PROD_DATABASE_NAME: str = os.getenv("PROD_DATABASE_NAME")
    
    # 실제 사용할 MongoDB 설정
    MONGODB_URI: str = PROD_MONGO_URI if ENV == "production" else DEV_MONGO_URI
    MONGODB_DB_NAME: str = PROD_DATABASE_NAME if ENV == "production" else DEV_DATABASE_NAME
    
    # API 키들
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
    GROK_API_KEY: str = os.getenv("GROK_API_KEY")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY")
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET")
    KAKAO_CLIENT_ID: str = os.getenv("KAKAO_CLIENT_ID")
    KAKAO_CLIENT_SECRET: str = os.getenv("KAKAO_CLIENT_SECRET")
    KAKAO_REDIRECT_URI: str = os.getenv("KAKAO_REDIRECT_URI")
    # KOSIS API key
    KOSIS_API_KEY: str = os.getenv("KOSIS_API_KEY")
    # WORK24 공통 코드 API 키
    WORK24_TRAINING_COMMON_API_KEY: str = os.getenv("WORK24_TRAINING_COMMON_API_KEY")
    # 국민내일배움카드 훈련과정 API 키
    WORK24_TOMORROW_API_KEY: str = os.getenv("WORK24_TOMORROW_API_KEY")
    #채용정보 API 키
    WORK24_RECRUIT_API_KEY: str = os.getenv("WORK24_RECRUIT_API_KEY")

    # 서버 설정
    SERVER_HOST: str = os.getenv("SERVER_HOST", "localhost")  # 개발환경은 localhost
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    API_VERSION: str = os.getenv("API_VERSION", "v1")
    
    # API 기본 URL - 환경에 따라 다르게 설정
    API_BASE_URL: str = (
        f"http://{SERVER_HOST}:{SERVER_PORT}/api/{API_VERSION}"
        if ENV == "production"
        else f"http://localhost:{SERVER_PORT}/api/{API_VERSION}"
    )
    
    # OpenAI 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    
    # 공공데이터포털 API 설정
    DATA_API_KEY: str = os.getenv("DATA_API_KEY")
    DATA_DECODING_KEY: str = os.getenv("DATA_DECODING_KEY")
    DATA_URL: str = os.getenv("DATA_URL")
    
    # GOV24 API 설정
    GOV24_API_KEY: str = os.getenv("GOV24_API_KEY")
    # 행정안전부_대한민국 공공서비스 정보 (보조금24)
    GOV24_BASE_URL: str = os.getenv("GOV24_BASE_URL")
    GOV24_SWAGGER_URL: str = os.getenv("GOV24_SWAGGER_URL")
    GOV24_ENCODING_KEY: str = os.getenv("GOV24_ENCODING_KEY")
    # KOSIS API URL
    KOSIS_API_URL: str = os.getenv("KOSIS_API_URL")

    # WORK24 공통 URL
    WORK24_COMMON_URL: str = os.getenv("WORK24_COMMON_URL")


    # WORK24 채용공고 검색 URL
    WORK24_RECRUIT_LIST_URL: str = os.getenv("WORK24_RECRUIT_LIST_URL")

    # 채용정보 상세 요청 URL
    WORK24_RECRUIT_DETAIL_URL: str = os.getenv("WORK24_RECRUIT_DETAIL_URL")
    # 채용정보 API 채용행사 목록 요청 URL
    WORK24_RECRUIT_URL: str = os.getenv("WORK24_RECRUIT_URL")
    # 채용정보 API 채용행사 상세 요청 URL
    WORK24_RECRUIT_DETAIL_URL: str = os.getenv("WORK24_RECRUIT_DETAIL_URL")
    # 채용정보 API 공채속보 목록 요청 URL
    WORK24_RECRUIT_ANNOUNCEMENT_URL: str = os.getenv("WORK24_RECRUIT_ANNOUNCEMENT_URL")
    # 채용정보 API 공채속보 상세 요청 URL
    WORK24_RECRUIT_ANNOUNCEMENT_DETAIL_URL: str = os.getenv("WORK24_RECRUIT_ANNOUNCEMENT_DETAIL_URL")
    # 채용정보 API 공채기업 목록 요청 URL
    WORK24_RECRUIT_COMPANY_URL: str = os.getenv("WORK24_RECRUIT_COMPANY_URL")
    # 채용정보 API 공채기업 상세 요청 URL
    WORK24_RECRUIT_COMPANY_DETAIL_URL: str = os.getenv("WORK24_RECRUIT_COMPANY_DETAIL_URL")

    # 국민내일배움카드 훈련과정 API 요청 URL
    WORK24_TM_URL: str = os.getenv("WORK24_TM_URL")
    WORK24_TM_INFO_URL: str = os.getenv("WORK24_TM_INFO_URL")
    WORK24_TM_SCHEDULE_URL: str = os.getenv("WORK24_TM_SCHEDULE_URL")

    # 공통 코드 API - 훈련 요청 URL
    WORK24_TRAINING_COMMON_URL: str = os.getenv("WORK24_TRAINING_COMMON_URL")
    # 공통 코드 API - 채용(지역,직종,자격면허,산업단지,지하철,전공,언어,학과계열,강소기업) 요청 URL
    # 지역
    WORK24_RECRUIT_LOC_URL: str = os.getenv("WORK24_RECRUIT_LOC_URL")
    # 직종
    WORK24_RECRUIT_JOB_URL: str = os.getenv("WORK24_RECRUIT_JOB_URL")
    # 자격면허
    WORK24_RECRUIT_LICENSE_URL: str = os.getenv("WORK24_RECRUIT_LICENSE_URL")
    # 산업단지
    WORK24_RECRUIT_INDUSTRY_URL: str = os.getenv("WORK24_RECRUIT_INDUSTRY_URL")
    # 지하철
    WORK24_RECRUIT_SUBWAY_URL: str = os.getenv("WORK24_RECRUIT_SUBWAY_URL")
    # 전공
    WORK24_RECRUIT_MAJOR_URL: str = os.getenv("WORK24_RECRUIT_MAJOR_URL")
    # 언어
    WORK24_RECRUIT_LANGUAGE_URL: str = os.getenv("WORK24_RECRUIT_LANGUAGE_URL")
    # 학과계열
    WORK24_RECRUIT_DEPARTMENT_URL: str = os.getenv("WORK24_RECRUIT_DEPARTMENT_URL")
    # 강소기업
    WORK24_RECRUIT_HIDDEN_CHAMPION_URL: str = os.getenv("WORK24_RECRUIT_HIDDEN_CHAMPION_URL")

    # CORS 설정 - 환경에 따라 다르게 설정
    ALLOWED_ORIGINS: List[str] = (
        ["http://54.180.157.200", "http://54.180.157.200:3000"]
        if ENV == "production"
        else ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
    )

    class Config:
        env_file = ".env"
        extra = "allow"  # 추가 필드 허용

settings = Settings()

# 환경 설정 로그 부분을 더 자세하게 수정
print("\n=== 환경 설정 상태 ===")
print(f"Current Environment: {settings.ENV}")
print(f"API Base URL: {settings.API_BASE_URL}")

print("\n=== MongoDB 설정 ===")
print(f"MongoDB URI: {settings.MONGODB_URI}")
print(f"MongoDB DB Name: {settings.MONGODB_DB_NAME}")

print("\n=== API 키 설정 ===")
print(f"OpenAI API Key: {'설정됨' if settings.OPENAI_API_KEY else '미설정'}")
print(f"OpenAI Model Name: {settings.OPENAI_MODEL_NAME}")
print(f"Google API Key: {'설정됨' if settings.GOOGLE_API_KEY else '미설정'}")
print(f"Grok API Key: {'설정됨' if settings.GROK_API_KEY else '미설정'}")
print(f"Work24 Recruit API Key: {'설정됨' if settings.WORK24_RECRUIT_API_KEY else '미설정'}")
print(f"Gov24 API Key: {'설정됨' if settings.GOV24_API_KEY else '미설정'}")

print("\n=== URL 설정 ===")
print(f"Work24 Common URL: {settings.WORK24_COMMON_URL}")
print(f"Data URL: {settings.DATA_URL}")

print("\n=== CORS 설정 ===")
print(f"Allowed Origins: {settings.ALLOWED_ORIGINS}")
print("==================\n") 