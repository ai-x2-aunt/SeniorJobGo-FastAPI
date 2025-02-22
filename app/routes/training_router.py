import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from app.models.schemas import TrainingSearchRequest

from app.agents.training_advisor import TrainingAdvisorAgent

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/trainings",
    tags=["trainings"]
)

def get_training_advisor(request: Request):
    """TrainingAdvisorAgent 인스턴스 가져오기"""
    return request.app.state.training_advisor

@router.post("/search")
async def search_trainings(
    search_params: TrainingSearchRequest,
    training_advisor: TrainingAdvisorAgent = Depends(get_training_advisor)
):
    """훈련정보 검색 API - 모달에서 직접 검색할 때 사용"""
    try:
        logger.info(f"[TrainingRouter] 훈련과정 검색 파라미터: {search_params}")

        # 지역 정보 처리
        location = search_params.location
        if not location and search_params.city:
            location = search_params.city
            if search_params.district:
                location += f" {search_params.district}"

        # 사용자 프로필 구성
        user_profile = {
            "location": location,
            "interests": search_params.interests,
            "preferredTime": search_params.preferredTime,
            "preferredDuration": search_params.preferredDuration
        }

        # NER 정보 구성
        user_ner = {
            "지역": location,
            "관심분야": search_params.interests,
            "직무": search_params.interests[0] if search_params.interests else ""
        }

        # 검색 쿼리 구성
        search_query = f"다음 조건에 맞는 훈련과정을 찾아주세요: "
        if location:
            search_query += f"지역: {location}, "
        if search_params.interests:
            search_query += f"관심분야: {', '.join(search_params.interests)}, "
        if search_params.preferredTime:
            search_query += f"선호시간: {search_params.preferredTime}, "
        if search_params.preferredDuration:
            search_query += f"선호기간: {search_params.preferredDuration}, "

        # TrainingAdvisor 호출
        result = await training_advisor.search_training_courses(
            query=search_query.rstrip(", "),
            user_profile=user_profile,
            user_ner=user_ner
        )

        # 응답 구성
        if not result.get("trainingCourses"):
            return {
                "message": "죄송합니다. 현재 조건에 맞는 훈련과정을 찾지 못했습니다.",
                "trainingCourses": []
            }
        return {
            "message": result.get("message", f"{location or '전국'} 지역의 훈련과정을 {len(result.get('trainingCourses', []))}건 찾았습니다."),
            "trainingCourses": result.get("trainingCourses", [])
        }

    except Exception as e:
        logger.error(f"[TrainingRouter] 처리 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"훈련과정 검색 중 오류가 발생했습니다: {str(e)}"
        )
