from fastapi import APIRouter, Request, Response, HTTPException, Depends
from pydantic import BaseModel
import logging
from app.models.schemas import ChatRequest, ChatResponse, JobPosting, TrainingCourse

from db.database import db
from bson import ObjectId
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.chat_agent import ChatAgent
from app.agents.flow_graph import build_flow_graph
from app.models.flow_state import FlowState
from app.agents.ner_extractor import extract_ner
from db.routes_auth import get_user_info_by_cookie
import os
import json
import time
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

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

async def format_chat_history(user_id: str, limit: int = 3) -> str:
    """MongoDB에서 최근 N개의 메시지만 가져와서 문자열로 변환"""
    try:
        # 최근 메시지만 가져오기
        chat_result = await db.users.aggregate([
            {"$match": {"_id": ObjectId(user_id)}},
            {"$project": {
                "messages": {"$slice": ["$messages", -limit]}  # 최근 limit개만 가져오기
            }}
        ]).to_list(1)
        
        if not chat_result:
            return ""
            
        messages = chat_result[0].get("messages", [])
        history = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history.append(f"User: {content}")
            elif role == "bot":
                history.append(f"Assistant: {content}")
                
        return "\n".join(history)
        
    except Exception as e:
        logger.error(f"대화 이력 조회 중 오류: {str(e)}")
        return ""

async def save_chat_message(user_id: str, message: Dict) -> bool:
    """새로운 메시지를 사용자의 대화 기록에 저장"""
    try:
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$push": {"messages": message}}
        )
        return True
    except Exception as e:
        logger.error(f"메시지 저장 중 오류: {str(e)}")
        return False

@router.post("/chat/", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """채팅 API"""
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
                type="error",
                error_message="사용자 정보를 찾을 수 없습니다."
            )

        # 최근 5개의 대화 이력만 가져오기
        chat_history = await format_chat_history(str(user["_id"]), limit=5)

        # NER 추출
        extracted_ner = await extract_ner(
            user_input=chat_request.user_message,
            llm=request.app.state.llm,
            user_profile=chat_request.user_profile
        )

        # 워크플로우 초기 상태 설정
        initial_state = FlowState(
            query=chat_request.user_message,
            chat_history=chat_history,  # 포맷된 chat_history 사용
            user_profile=chat_request.user_profile or {},
            agent_type="",
            agent_reason="",
            tool_response=None,
            final_response={},
            error_message="",
            jobPostings=[],
            trainingCourses=[],
            policyPostings=[],
            mealPostings=[],
            messages=[],
            user_ner=extracted_ner,
            supervisor=request.app.state.supervisor  # Supervisor 인스턴스 전달
        )

        # 워크플로우 생성
        workflow = await build_flow_graph(llm=request.app.state.llm)

        # 워크플로우 실행
        final_state = await workflow.ainvoke(
            initial_state,
            {"configurable": {"thread_id": str(user["_id"])}}
        )
        
        logger.info(f"[ChatRouter] 워크플로우 결과 - State: {final_state}")
        
        # 최종 응답 변환
        try:
            # AddableValuesDict -> dict 변환
            if hasattr(final_state, "value"):
                final_state_dict = final_state.value
            else:
                final_state_dict = final_state

            response = ChatResponse(
                message=final_state_dict.get("final_response", {}).get("message", ""),
                type=final_state_dict.get("final_response", {}).get("type", "chat"),
                jobPostings=final_state_dict.get("jobPostings", []),
                trainingCourses=final_state_dict.get("trainingCourses", []),
                policyPostings=final_state_dict.get("policyPostings", []),
                mealPostings=final_state_dict.get("mealPostings", []),
                user_profile=chat_request.user_profile or {}
            )
            
            logger.info(f"[ChatRouter] 최종 응답: {response}")

            # 사용자 메시지 저장
            user_message = {
                "role": "user",
                "content": chat_request.user_message,
                "timestamp": datetime.now(),
                "type": "chat"
            }
            await save_chat_message(user["_id"], user_message)
            
            # 응답 생성 후
            bot_message = {
                "role": "bot",
                "content": response.message,
                "timestamp": datetime.now(),
                "type": response.type,
                "metadata": {
                    "jobPostings": response.jobPostings,
                    "trainingCourses": response.trainingCourses,
                    "policyPostings": response.policyPostings,
                    "mealPostings": response.mealPostings
                }
            }
            await save_chat_message(user["_id"], bot_message)
            
            return response

        except Exception as parse_error:
            logger.error(f"[ChatRouter] 응답 변환 중 오류: {str(parse_error)}")
            logger.error(f"[ChatRouter] final_state 타입: {type(final_state)}")
            logger.error(f"[ChatRouter] final_state 내용: {final_state}")
            
            return ChatResponse(
                message="죄송합니다. 응답 처리 중 문제가 발생했습니다.",
                type="error",
                jobPostings=[],
                trainingCourses=[],
                policyPostings=[],
                mealPostings=[],
                user_profile=chat_request.user_profile or {}
            )
        
    except Exception as e:
        logger.error(f"[ChatRouter] 처리 중 오류: {str(e)}", exc_info=True)
        return ChatResponse(
            message="죄송합니다. 응답 처리 중 문제가 발생했습니다.",
            type="error",
            jobPostings=[],
            trainingCourses=[],
            policyPostings=[],
            mealPostings=[],
            user_profile=chat_request.user_profile or {}
        )

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
            "pageSize": 100,        # 검색 결과 수
            "outType": "1",        # 리스트 형태
            "sort": "ASC",        # 최신순
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
@router.post("/policy/search", response_model=ChatResponse)
async def search_policies(request: ChatRequest):
    try:
        from app.agents.policy_agent import query_policy_agent
        logger.info(f"[PolicySearch] 정책 검색 요청: {request.user_message}")
        
        result = await query_policy_agent(request.user_message)
        logger.info(f"[PolicySearch] 검색 결과: {result}")
        
        # ChatResponse 형식으로 변환
        response = ChatResponse(
            message=result.get("message", ""),
            type="policy",
            jobPostings=[],
            trainingCourses=[],
            policyPostings=result.get("policyPostings", []),
            mealPostings=[],
            user_profile=request.user_profile or {}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"[PolicySearch] 오류 발생: {str(e)}")
        return ChatResponse(
            message="정책 검색 중 오류가 발생했습니다.",
            type="error",
            jobPostings=[],
            trainingCourses=[],
            policyPostings=[],
            mealPostings=[],
            user_profile=request.user_profile or {}
        )