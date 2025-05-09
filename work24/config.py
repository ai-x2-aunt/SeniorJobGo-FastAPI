from pathlib import Path
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(override=True)

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_DIR = BASE_DIR / "work24" / "training_posting"

# API 기본 URL
WORK24_COMMON_URL = os.getenv("WORK24_COMMON_URL")

# API 설정
TRAINING_APIS = {
    "training_common": {
        "name": "공통코드",
        "api_key": os.getenv("WORK24_TRAINING_COMMON_API_KEY"),
        "endpoints": {
            "common": os.getenv("WORK24_TRAINING_COMMON_URL")
        }
    },
    "tomorrow": {
        "name": "국민내일배움카드",
        "api_key": os.getenv("WORK24_TOMORROW_API_KEY"),
        "endpoints": {
            "list": os.getenv("WORK24_TM_URL"),
            "info": os.getenv("WORK24_TM_INFO_URL"),
            "schedule": os.getenv("WORK24_TM_SCHEDULE_URL")
        }
    },
    "business": {
        "name": "사업주훈련",
        "api_key": os.getenv("WORK24_BUSINESS_API_KEY"),
        "endpoints": {
            "list": os.getenv("WORK24_BUSINESS_URL"),
            "info": os.getenv("WORK24_BUSINESS_DETAIL_URL"),
            "schedule": os.getenv("WORK24_BUSINESS_SCHEDULE_URL")
        }
    },
    "consortium": {
        "name": "국가인적자원개발 컨소시엄",
        "api_key": os.getenv("WORK24_CONSORTIUM_API_KEY"),
        "endpoints": {
            "list": os.getenv("WORK24_CONSORTIUM_URL"),
            "info": os.getenv("WORK24_CONSORTIUM_DETAIL_URL"),
            "schedule": os.getenv("WORK24_CONSORTIUM_SCHEDULE_URL")
        }
    },
    "parallel": {
        "name": "일학습병행",
        "api_key": os.getenv("WORK24_PARALLEL_API_KEY"),
        "endpoints": {
            "list": os.getenv("WORK24_PARALLEL_URL"),
            "info": os.getenv("WORK24_PARALLEL_DETAIL_URL"),
            "schedule": os.getenv("WORK24_PARALLEL_SCHEDULE_URL")
        }
    }
}

# 파일 저장 설정
JSON_FILENAME_FORMAT = "{api_type}_{endpoint}_{timestamp}.json" 