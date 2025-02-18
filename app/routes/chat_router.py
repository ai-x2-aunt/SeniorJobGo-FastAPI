from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
from app.models.schemas import ChatRequest, ChatResponse, JobPosting, TrainingCourse

from db.database import db
from db.routes_auth import get_user_info_by_cookie
from db.routes_chat import add_chat_message
from bson import ObjectId
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.chat_agent import ChatAgent

import os
import json
import time
from typing import Optional, Dict, Any
from app.agents.policy_agent import query_policy_agent


logger = logging.getLogger(__name__)

router = APIRouter()

# 정책 검색 요청 모델 정의
class PolicySearchRequest(BaseModel):
    """정책 검색 요청 모델"""
    user_message: str
    user_profile: Dict = {}
    session_id: Optional[str] = None

# 이전 대화 내용을 저장할 딕셔너리
conversation_history = {}

# 의존성 함수들
def get_llm(request: Request):
    return request.app.state.llm

def get_vector_search(request: Request):
    return request.app.state.vector_search

def get_job_advisor_agent(request: Request):
    return request.app.state.job_advisor_agent

def get_chat_agent(request: Request, llm=Depends(get_llm)):
    return ChatAgent(llm=llm)

@router.post("/chat/", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    job_advisor_agent: JobAdvisorAgent = Depends(get_job_advisor_agent)
) -> ChatResponse:
    start_time = time.time()
    try:
        # 요청 시작 로깅
        logger.info("="*50)
        logger.info("[ChatRouter] 새로운 채팅 요청 시작")
        logger.info(f"[ChatRouter] 요청 메시지: {chat_request.user_message}")
        logger.info(f"[ChatRouter] 사용자 프로필: {chat_request.user_profile}")
        
        # DB 체크
        if db is None:
            logger.error("[ChatRouter] DB 연결 없음")
            raise Exception("DB connection is None")

        # 쿠키에서 사용자 정보 조회
        user = None
        try:
            user = await get_user_info_by_cookie(request)
        except:
            logger.error(f"[ChatRouter] 쿠키에서 사용자 정보 조회 중 오류 발생 - 기본 응답 반환")
            return ChatResponse(
                message=chat_request.user_message,
                jobPostings=[],
                trainingCourses=[],
                type="info",
                user_profile=chat_request.user_profile or {}
            )

        # 이전 대화 이력 가져오기
        chat_history = user.get("messages", [])
        
        # 채용/훈련 정보 관련 대화만 필터링
        formatted_history = ""
        if chat_history:  # chat_history가 있을 때만 필터링 수행
            for i in range(len(chat_history)-1, -1, -1):  # 최신 메시지부터 역순으로 확인
                msg = chat_history[i]
                if isinstance(msg, dict):  # dict 타입 체크 추가
                    role = "사용자" if msg.get("role") == "user" else "시스템"
                    content = msg.get("content", "")
                    # dict인 경우 필요한 정보만 추출
                    if isinstance(content, dict):
                        content = content.get("message", "")
                    elif not isinstance(content, str):
                        content = str(content)
                    formatted_history = f"{role}: {content}\n" + formatted_history

        # 이전 검색 결과가 있는지 확인 (빈 대화 이력 처리)
        prev_message = ""
        if formatted_history:
            history_lines = formatted_history.strip().split("\n")
            for line in reversed(history_lines):
                if line.startswith("시스템:"):
                    prev_message = line.replace("시스템:", "").strip()
                    break

        # 이전 검색에서 훈련과정 관련 키워드가 있는지 확인
        if prev_message and "관련 훈련과정을" in prev_message and any(word in chat_request.user_message for word in ["저렴한", "저렴하게", "싼", "무료로", "비용이", "학비가"]):
            # 이전 검색 키워드 추출
            keyword = prev_message.split("'")[1] if "'" in prev_message else ""
            if keyword:
                chat_request.user_message = f"{keyword} {chat_request.user_message}"
        
        logger.info("[ChatRouter] JobAdvisorAgent.chat 호출 시작")
        try:
            response = await job_advisor_agent.chat(
                query=chat_request.user_message,
                user_profile=chat_request.user_profile,
                chat_history=formatted_history
            )
            logger.info("[ChatRouter] JobAdvisorAgent.chat 응답 성공")
            logger.info(f"[ChatRouter] 응답 내용: {response}")
            
        except Exception as chat_error:
            logger.error("[ChatRouter] JobAdvisorAgent.chat 실행 중 에러", exc_info=True)
            raise chat_error
        
        # 가격 필터링이 필요한 경우
        if response.get("type") == "training" and any(word in chat_request.user_message for word in ["저렴한", "저렴하게", "싼", "무료로", "비용이", "학비가"]):
            courses = response.get("trainingCourses", [])
            filtered_courses = []
            for course in courses:
                try:
                    cost = int(course.get("cost", "0").replace(",", "").replace("원", ""))
                    if cost <= 300000:  # 30만원 이하
                        filtered_courses.append(course)
                except ValueError:
                    continue
            
            if filtered_courses:
                response["trainingCourses"] = filtered_courses
                response["message"] = f"30만원 이하의 저렴한 훈련과정 {len(filtered_courses)}개를 찾았습니다."
            else:
                response["message"] = "죄송합니다. 조건에 맞는 저렴한 훈련과정을 찾지 못했습니다."

        # 메시지 저장
        try:
            await add_chat_message(user, chat_request.user_message, response.get("message", ""))
            logger.info("[ChatRouter] 메시지 저장 완료")
        except:
            logger.error("[ChatRouter] 메시지 저장 중 에러", exc_info=True)
            # 메시지 저장 실패는 치명적이지 않으므로 계속 진행

        # 응답 생성
        processing_time = time.time() - start_time
        response["processingTime"] = round(processing_time, 2)

        # JobPosting 목록 생성
        job_postings_list = []
        for idx, jp in enumerate(response.get("jobPostings", [])):
            job_postings_list.append(JobPosting(
                id=jp.get("id", "no_id"),
                location=jp.get("location", ""),
                company=jp.get("company", ""),
                title=jp.get("title", ""),
                salary=jp.get("salary", ""),
                workingHours=jp.get("workingHours", "정보없음"),
                description=jp.get("description", ""),
                phoneNumber=jp.get("phoneNumber", ""),
                deadline=jp.get("deadline", ""),
                requiredDocs=jp.get("requiredDocs", ""),
                hiringProcess=jp.get("hiringProcess", ""),
                insurance=jp.get("insurance", ""),
                jobCategory=jp.get("jobCategory", ""),
                jobKeywords=jp.get("jobKeywords", ""),
                posting_url=jp.get("posting_url", ""),
                rank=jp.get("rank", idx+1)
            ))

        # TrainingCourse 목록 생성
        training_courses_list = []
        for course in response.get("trainingCourses", []):
            training_courses_list.append(TrainingCourse(
                id=course.get("id", ""),
                title=course.get("title", ""),
                institute=course.get("institute", ""),
                location=course.get("location", ""),
                period=course.get("period", ""),
                startDate=course.get("startDate", ""),
                endDate=course.get("endDate", ""),
                cost=course.get("cost", ""),
                description=course.get("description", ""),
                target=course.get("target"),
                yardMan=course.get("yardMan"),
                titleLink=course.get("titleLink"),
                telNo=course.get("telNo")
            ))

        # 최종 응답 생성
        response_model = ChatResponse(
            message=response.get("message", ""),
            jobPostings=job_postings_list,
            trainingCourses=training_courses_list,
            type=response.get("type", "info"),
            user_profile=response.get("user_profile", {})
        )
        
        logger.info("[ChatRouter] 응답 생성 완료")
        logger.info("="*50)
        
        return response_model

    except Exception as e:
        logger.error("[ChatRouter] 치명적인 에러 발생", exc_info=True)
        error_response = ChatResponse(
            type="error",
            message="처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            jobPostings=[],
            trainingCourses=[],
            user_profile={}
        )
        return error_response

# 훈련정보 검색 엔드포인트 추가
@router.post("/training/search")
async def search_training(
    chat_request: ChatRequest,
    job_advisor_agent: JobAdvisorAgent = Depends(get_job_advisor_agent)
):
    try:
        # 대화 문맥 확인
        context = chat_request.context if hasattr(chat_request, 'context') else {}
        mode = context.get('mode', 'training')
        
        if mode != 'training':
            return {
                "message": "훈련정보 검색을 위해 기본 정보를 입력해주세요.",
                "trainingCourses": [],
                "type": "info"
            }

        # work24 API에서 훈련정보 가져오기
        from work24.training_collector import TrainingCollector
        collector = TrainingCollector()
        
        # 사용자 프로필에서 검색 조건 추출
        user_profile = chat_request.user_profile or {}
        
        # API 검색 파라미터 설정
        search_params = {
            "srchTraProcessNm": "",  # 기본값 빈 문자열
            "srchTraArea1": "11",  # 기본 서울
            "srchTraArea2": "",    # 상세 지역
            "srchTraStDt": "",     # 시작일
            "srchTraEndDt": "",    # 종료일
            "pageSize": 20,        # 검색 결과 수
            "outType": "1",        # 리스트 형태
            "sort": "DESC",        # 최신순
            "sortCol": "TRNG_BGDE" # 훈련시작일 기준
        }
        
        # 지역 정보 처리 개선
        location = user_profile.get("location", "")
        if location:
            area_codes = {
                "서울": "11", "경기": "41", "인천": "28",
                "부산": "26", "대구": "27", "광주": "29",
                "대전": "30", "울산": "31", "세종": "36",
                "강원": "42", "충북": "43", "충남": "44",
                "전북": "45", "전남": "46", "경북": "47",
                "경남": "48", "제주": "50"
            }
            
            # 지역명에서 시/도 추출
            for area, code in area_codes.items():
                if area in location:
                    search_params["srchTraArea1"] = code
                    break
                    
            # 상세 지역 처리
            if "서울" in location:
                districts = {
                    "강남구": "GN", "강동구": "GD", "강북구": "GB",
                    "강서구": "GS", "관악구": "GA", "광진구": "GJ",
                    "구로구": "GR", "금천구": "GC", "노원구": "NW",
                    "도봉구": "DB", "동대문구": "DD", "동작구": "DJ",
                    "마포구": "MP", "서대문구": "SD", "서초구": "SC",
                    "성동구": "SD", "성북구": "SB", "송파구": "SP",
                    "양천구": "YC", "영등포구": "YD", "용산구": "YS",
                    "은평구": "EP", "종로구": "JR", "중구": "JG",
                    "중랑구": "JL"
                }
                for district, code in districts.items():
                    if district in location:
                        search_params["srchTraArea2"] = code
                        break
                
        # 관심분야 키워드 매핑 개선
        interests = user_profile.get("interests", "").lower()
        if interests:
            keyword_mapping = {
                "it": ["정보", "IT", "컴퓨터", "프로그래밍", "소프트웨어"],
                "요양": ["요양", "복지", "간호", "의료", "보건"],
                "조리": ["조리", "요리", "식품", "외식", "주방"],
                "사무": ["사무", "행정", "경영", "회계", "총무"],
                "서비스": ["서비스", "고객", "판매", "영업", "상담"],
                "제조": ["제조", "생산", "가공", "기계", "설비"]
            }
            
            keywords = []
            for category, category_keywords in keyword_mapping.items():
                if any(kw in interests for kw in [category, *category_keywords]):
                    keywords.extend(category_keywords)
            
            if keywords:
                search_params["srchTraProcessNm"] = ",".join(set(keywords))
        elif chat_request.user_message:
            # 사용자 메시지에서 키워드 추출 개선
            message_keywords = [
                word for word in chat_request.user_message.split() 
                if len(word) >= 2 and not any(c in word for c in ".,?! ")
            ]
            if message_keywords:
                search_params["srchTraProcessNm"] = ",".join(message_keywords[:3])
        
        # Work24 API 호출
        courses = collector._fetch_training_list("tomorrow", search_params)
        
        if not courses:
            return {
                "message": "현재 조건에 맞는 훈련과정이 없습니다. 다른 조건으로 찾아보시겠어요?",
                "trainingCourses": [],
                "type": "info"
            }
            
        # 최대 5개 과정만 반환
        top_courses = courses[:5]
        
        # 응답 데이터 구성
        training_courses = []
        for course in top_courses:
            training_courses.append({
                "id": course.get("trprId", ""),
                "title": course.get("title", ""),
                "institute": course.get("subTitle", ""),
                "location": course.get("address", ""),
                "period": f"{course.get('traStartDate', '')} ~ {course.get('traEndDate', '')}",
                "startDate": course.get("traStartDate", ""),
                "endDate": course.get("traEndDate", ""),
                "cost": f"{int(course.get('courseMan', 0)):,}원",
                "description": course.get("contents", ""),
                "target": course.get("trainTarget", ""),
                "ncsCd": course.get("ncsCd", ""),
                "yardMan": course.get("yardMan", ""),
                "titleLink": course.get("titleLink", ""),
                "telNo": course.get("telNo", "")
            })
            
        return {
            "message": f"'{chat_request.user_message}' 검색 결과, {len(training_courses)}개의 훈련과정을 찾았습니다.",
            "trainingCourses": training_courses,
            "type": "training",
            "context": {
                "mode": "training",
                "lastQuery": chat_request.user_message,
                "userProfile": user_profile
            }
        }
        
    except Exception as e:
        logger.error(f"Training search error: {str(e)}")
        logger.error("상세 에러:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# policy
@router.post("/policy/search")
async def search_policies(request: PolicySearchRequest):
    try:
        logger.info(f"[PolicySearch] 정책 검색 요청: {request.user_message}")
        result = query_policy_agent(request.user_message)
        logger.info(f"[PolicySearch] 검색 결과: {result}")
        return result
    except Exception as e:
        logger.error(f"[PolicySearch] 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="정책 검색 중 오류가 발생했습니다."
        )