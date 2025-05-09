from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
import logging
import json
from app.agents.policy_agent import query_policy_agent
from app.services.meal_data_client import PublicDataClient
from app.core.config import settings

logger = logging.getLogger(__name__)

###############################################################################
# SupervisorAgent (ReAct 사용)
###############################################################################

def build_supervisor_agent() -> AgentExecutor:
    """SupervisorAgent(ReAct) 생성"""
    
    # LLM 설정
    llm = ChatOpenAI(
        model_name=settings.OPENAI_MODEL_NAME,
        temperature=0.5
    )
    
    # 사용할 Tool 정의
    tools: List[BaseTool] = [
        Tool(
            name="job_advisor_tool",
            func=job_advisor_tool_func,
            description="채용 정보 검색 및 추천. 채용, 일자리, 구인 등과 관련된 질문에 사용",
            coroutine=job_advisor_tool_func  # 비동기 함수 지정
        ),
        Tool(
            name="training_advisor_tool",
            func=training_advisor_tool_func,
            description="교육/훈련 과정 검색 및 추천. 교육, 훈련, 강좌 등과 관련된 질문에 사용",
            coroutine=training_advisor_tool_func  # 비동기 함수 지정
        ),
        Tool(
            name="chat_agent_tool",
            func=chat_agent_tool_func,
            description="일반적인 대화 처리. 맛집, 날씨, 교통 등 일상적인 질문에 사용",
            coroutine=chat_agent_tool_func  # 비동기 함수 지정
        ),
        Tool(
            name="meal_agent_tool",
            func=meal_agent_tool_func,
            description="무료급식소 정보 검색. 무료급식, 급식소, 식사 등과 관련된 질문에 사용",
            coroutine=meal_agent_tool_func  # 비동기 함수 지정
        ),
        # 새로운 Tool 추가
        Tool(
            name="policy_advisor_tool",
            func=policy_advisor_tool_func,
            description="고령자 관련 정책 정보 검색. 정부지원, 복지, 연금 등과 관련된 질문에 사용",
            coroutine=policy_advisor_tool_func  # 비동기 함수 지정
        )
    ]

    # ReAct 프롬프트 템플릿
    react_template = """당신은 고령자를 위한 채용/교육/무료급식 상담 시스템의 관리자입니다.
사용자의 질문을 분석하여 필요한 도구들을 순차적으로 사용해 정보를 얻고, 종합적인 답변을 생성하세요.

사용자 입력: {input}

사용 가능한 도구들:
{tools}

You can call multiple tools using exactly this format:
Action: {tool_names}
Action Input: 입력은 반드시 아래 형식의 JSON으로 작성하세요
{{
    "query": "사용자 질문",
    "user_profile": "사용자 프로필 정보",
    "user_ner": "NER 추출 정보"
}}

Use exactly the ReAct format:
Question: {input}
Thought: {agent_scratchpad}

규칙:
1. 입력된 JSON에서 query, user_profile, user_ner 정보를 모두 활용하여 판단하세요.
2. 도구는 한 번에 하나씩만 호출하세요
3. 여러 정보가 필요한 경우 도구를 순차적으로 호출하세요
4. 각 도구의 결과를 모아서 최종 답변을 생성하세요
5. Action Input은 반드시 JSON 형식으로 작성하세요
"""

    # PromptTemplate으로 변환
    prompt = PromptTemplate.from_template(react_template)

    # ReAct Agent 생성
    react_agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        stop_sequence="final_answer"  # Final Answer에서 멈추도록
    )

    # AgentExecutor로 감싸서 반환
    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=2,  # 반복 횟수 더 제한
        max_execution_time=None,  # 시간 제한 제거
        early_stopping_method="force",  # generate -> force로 변경
        return_intermediate_steps=True,  # 중간 단계 결과 반환
        verbose=True  # 디버깅을 위한 상세 로그
    )

class SupervisorAgent:
    """ReAct 기반 관리자 에이전트"""
    
    def __init__(self, llm: ChatOpenAI):
        self.agent = build_supervisor_agent()
        self.data_client = PublicDataClient()
    
    async def analyze_query(self, user_input: str, user_profile: dict = None, chat_history: str = "", user_ner: dict = None) -> dict:
        """사용자 입력 분석"""
        try:
            # 입력 데이터 구성
            input_data = {
                "query": user_input,
                "user_profile": user_profile or {},
                "chat_history": chat_history,
                "user_ner": user_ner or {}
            }
            
            # ReAct 실행 - async 방식으로 변경
            result = await self.agent.ainvoke(json.dumps(input_data, ensure_ascii=False))
            return result
            
        except Exception as e:
            logger.error(f"[SupervisorAgent] 분석 중 오류: {str(e)}")
            return {
                "message": f"처리 중 오류 발생: {str(e)}",
                "type": "error"
            }

    def is_satisfied(self, tool_response: dict) -> bool:
        """
        Tool(또는 Agent) 결과가 충분히 만족스러운지 여부를 ReAct로 물어볼 수 있다고 가정.
        예시로만 작성.
        """
        try:
            check_str = f"이 응답이 충분한지 판단해줘: {tool_response}"
            result = self.agent.invoke(json.dumps(check_str, ensure_ascii=False))
            return "yes" in result.lower()
        except Exception as e:
            logger.error(f"[SupervisorAgent] is_satisfied 에러: {str(e)}", exc_info=True)
            return True  # 실패 시 일단 True로
        
   ###############################################################################
# 예시 Job / Training / Chat 처리용 함수 (각 Tool이 실제로 호출)
###############################################################################

async def job_advisor_tool_func(input_str: str) -> str:
    """채용 정보 검색 도구"""
    try:
        # 입력값 검증 및 파싱
        if not input_str:
            raise ValueError("입력값이 비어있습니다")
            
        # 입력값이 문자열인지 확인하고 파싱
        if not isinstance(input_str, str):
            input_str = json.dumps(input_str, ensure_ascii=False)
            
        try:
            data = json.loads(input_str)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 문자열을 직접 딕셔너리로 변환 시도
            data = {"query": input_str, "user_profile": {}}
        
        logger.info(f"[job_advisor_tool] input_str: {input_str}")

        query = data.get("query", "")
        user_profile = data.get("user_profile", {})
        user_ner = data.get("user_ner", {})
        chat_history = data.get("chat_history", "")
        
        logger.info(f"[job_advisor_tool] 입력 데이터: {data}")
        
        # job_advisor 호출
        from app.main import app
        job_advisor = app.state.job_advisor
        response = await job_advisor.chat(
            query=query,
            user_profile=user_profile,
            user_ner=user_ner,
            chat_history=[]  # 빈 리스트로 전달
        )
        
        # 응답이 이미 dict인 경우 JSON으로 변환
        if isinstance(response, dict):
            # type을 job으로 통일
            response["type"] = "job"
            return json.dumps(response, ensure_ascii=False)
            
        # 문자열 응답인 경우 기본 형식으로 변환
        return json.dumps({
            "message": str(response),
            "type": "job",
            "jobPostings": [],
            "final_answer": str(response)
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[job_advisor_tool] 오류: {str(e)}", exc_info=True)
        return json.dumps({
            "message": f"채용정보 검색 중 오류: {str(e)}",
            "type": "error",
            "jobPostings": []
        }, ensure_ascii=False)


async def training_advisor_tool_func(input_str: str) -> str:
    """교육/훈련 과정 검색 도구"""
    try:
        # 입력값 검증 및 파싱
        if not input_str:
            raise ValueError("입력값이 비어있습니다")
            
        # 여러 줄 JSON 문자열 처리
        if isinstance(input_str, str):
            # 줄바꿈과 공백 제거
            input_str = input_str.strip().replace('\n', '')
            
        try:
            data = json.loads(input_str)
        except json.JSONDecodeError:
            data = {"query": input_str, "user_profile": {}}
        
        logger.info(f"[training_advisor_tool] input_str: {input_str}")

        query = data.get("query", "")
        user_profile = data.get("user_profile", {})
        user_ner = data.get("user_ner", {})
        chat_history = data.get("chat_history", "")
        
        # training_advisor 호출
        from app.main import app
        training_advisor = app.state.training_advisor
        
        # search_training_courses 메서드 호출 (chat 메서드 대신)
        response = await training_advisor.search_training_courses(
            query=query,
            user_profile=user_profile,
            user_ner=user_ner,
            chat_history=[]
        )
        
        # 응답이 이미 dict인 경우 JSON으로 변환
        if isinstance(response, dict):
            response["type"] = "training"  # type 통일
            return json.dumps(response, ensure_ascii=False)
            
        # 문자열 응답인 경우 기본 형식으로 변환
        return json.dumps({
            "message": str(response),
            "type": "training",
            "trainingCourses": [],
            "final_answer": str(response)
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[training_advisor_tool] 오류: {str(e)}", exc_info=True)
        return json.dumps({
            "message": f"훈련과정 검색 중 오류 발생: {str(e)}",
            "type": "error",
            "trainingCourses": []
        }, ensure_ascii=False)


async def chat_agent_tool_func(input_str: str) -> str:
    """일반 대화 처리 도구"""
    try:
        # 입력값 검증 및 파싱
        if not input_str:
            raise ValueError("입력값이 비어있습니다")
            
        # 입력값이 문자열인지 확인하고 파싱
        if not isinstance(input_str, str):
            input_str = json.dumps(input_str, ensure_ascii=False)
            
        # 입력값이 JSON 형식인지 확인
        try:
            data = json.loads(input_str)
            query = data.get("query", input_str)
            user_profile = data.get("user_profile", {})
        except json.JSONDecodeError:
            # JSON이 아닌 경우 문자열 그대로 사용
            query = input_str
            user_profile = {}
            
        # FastAPI app state에서 chat_agent 가져오기
        from app.main import app
        chat_agent = app.state.chat_agent
        
        # await를 사용하여 비동기 함수 호출
        response = await chat_agent.chat(query, user_profile)
        
        # 응답이 이미 dict인 경우 JSON으로 변환
        if isinstance(response, dict):
            return json.dumps(response, ensure_ascii=False)
            
        # 문자열 응답인 경우 기본 형식으로 변환
        return json.dumps({
            "message": str(response),
            "type": "chat",
            "final_answer": str(response)
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[chat_agent_tool] 오류: {str(e)}", exc_info=True)
        return json.dumps({
            "message": f"대화 처리 중 오류 발생: {str(e)}",
            "type": "error"
        }, ensure_ascii=False)

async def policy_advisor_tool_func(input_str: str) -> str:
    """정책 정보 검색 도구"""
    try:
        # 입력값 검증 및 파싱
        if not input_str:
            raise ValueError("입력값이 비어있습니다")
            
        # 입력값이 문자열인지 확인하고 파싱
        if not isinstance(input_str, str):
            input_str = json.dumps(input_str, ensure_ascii=False)
            
        try:
            data = json.loads(input_str)
            query = data.get("query", input_str)
            user_profile = data.get("user_profile", {})
            user_ner = data.get("user_ner", {})
        except json.JSONDecodeError:
            query = input_str
            
        # policy_agent 호출
        response = await query_policy_agent(query)
        
        # 응답이 이미 dict인 경우 JSON으로 변환
        if isinstance(response, dict):
            policies = response.get("policyPostings", [])
            
            if policies:
                message = f"총 {len(policies)}건의 정책 정보를 찾았습니다"
            else:
                message = "죄송합니다. 현재 조건에 맞는 정책 정보를 찾지 못했습니다"

            return json.dumps({
                "message": message,  # 직접 생성한 메시지 사용
                "type": "policy",
                "policyPostings": policies,
                "final_answer": message  # message와 동일한 값 사용
            }, ensure_ascii=False)
            
        # 문자열 응답인 경우 기본 형식으로 변환
        return json.dumps({
            "message": str(response),
            "type": "policy", 
            "policyPostings": [],
            "final_answer": str(response)
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[policy_advisor_tool] 오류: {str(e)}", exc_info=True)
        return json.dumps({
            "message": f"정책 정보 검색 중 오류: {str(e)}",
            "type": "error",
            "policyPostings": [],
            "final_answer": "정책 정보 검색 중 오류가 발생했습니다."
        }, ensure_ascii=False)

async def meal_agent_tool_func(input_str: str) -> str:
    """무료급식소 검색 도구"""
    try:
        # 입력값 검증 및 파싱
        if not input_str:
            raise ValueError("입력값이 비어있습니다")
            
        # 입력값이 문자열인지 확인하고 파싱
        if not isinstance(input_str, str):
            input_str = json.dumps(input_str, ensure_ascii=False)
            
        try:
            data = json.loads(input_str)
            query = data.get("query", input_str)
            user_profile = data.get("user_profile", {})
            user_ner = data.get("user_ner", {})
        except json.JSONDecodeError:
            query = input_str
            user_ner = {}
        
        logger.info(f"[meal_agent_tool] input_str: {input_str}")

        region = user_ner.get("지역", "")
        
        # data_client 인스턴스 생성
        data_client = PublicDataClient()
        
        # 전체 데이터 가져오기
        all_services = data_client.fetch_meal_services()
        
        # 지역 필터링
        if region:
            filtered_services = data_client.filter_by_region(all_services, region)
        else:
            filtered_services = all_services[:5]  # 지역이 없으면 상위 5개만
            
        # 결과 포맷팅
        meal_services = []
        for service in filtered_services[:5]:  # 상위 5개만
            meal_services.append({
                "name": service.get("fcltyNm", ""),  # API 필드명에 맞게 수정
                "address": service.get("rdnmadr", ""),
                "phone": service.get("phoneNumber", ""),
                "operatingHours": service.get("mlsvTime", ""),
                "targetGroup": service.get("mlsvTrget", ""),
                "description": service.get("mlsvDate", ""),
                "latitude": float(service.get("latitude", "0.0") or "0.0"),    # 안전한 형변환
                "longitude": float(service.get("longitude", "0.0") or "0.0")   # 안전한 형변환
            })

        # 사용자 친화적인 메시지 생성
        if meal_services:
            message = f"{region or '전국'}의 무료급식소 {len(meal_services)}곳을 찾았습니다"
        else:
            message = f"{region or '전국'}에서 무료급식소를 찾을 수 없습니다"

        return json.dumps({
            "message": message,
            "type": "meal",
            "mealPostings": meal_services[:5],  # 변환된 데이터 사용
            "final_answer": message
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"[meal_agent_tool] 오류: {str(e)}", exc_info=True)
        return json.dumps({
            "message": f"무료급식소 검색 중 오류 발생: {str(e)}",
            "type": "error",
            "mealPostings": [],
            "final_answer": "무료급식소 정보 검색 중 오류가 발생했습니다."
        }, ensure_ascii=False)