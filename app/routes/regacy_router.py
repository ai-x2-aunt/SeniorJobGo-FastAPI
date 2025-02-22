
from fastapi import APIRouter
import logging
import json
from app.core.prompts import EXTRACT_INFO_PROMPT

router = APIRouter()

logger = logging.getLogger(__name__)

# 채용 공고 추천 프롬프트 (레거시 코드)
@router.post("/api/v1/extract_info/")
async def extract_user_info(request: dict):
    try:
        user_message = request.get("user_message", "")
        chat_history = request.get("chat_history", "")
        
        logger.info(f"[extract_info] 요청 메시지: {user_message}")
        logger.info(f"[extract_info] 대화 이력: {chat_history}")
        
        # LLM 호출
        response = router.state.llm.invoke(
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