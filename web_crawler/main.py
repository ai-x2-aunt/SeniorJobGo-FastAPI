import os
from dotenv import load_dotenv
from .agents import create_crawler_agent
from .config import SEARCH_KEYWORDS

def main():
    # 환경 변수 로드
    load_dotenv(override=True)
    
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    # 크롤러 에이전트 생성
    run_crawler = create_crawler_agent()
    
    # 크롤러 실행
    print("채용 정보 크롤링을 시작합니다...")
    run_crawler(SEARCH_KEYWORDS)
    print("크롤링이 완료되었습니다.")

if __name__ == "__main__":
    main() 