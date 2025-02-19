import logging
import os
import json
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.schema.runnable import Runnable
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from app.core.prompts import verify_prompt, rewrite_prompt, generate_prompt, chat_prompt, EXTRACT_INFO_PROMPT, CLASSIFY_INTENT_PROMPT, RESUME_GUIDE_PROMPT, RESUME_FEEDBACK_PROMPT

from app.agents.chat_agent import ChatAgent
from app.agents.training_advisor import TrainingAdvisorAgent
from app.services.vector_store_search import VectorStoreSearch
from app.services.document_filter import DocumentFilter
from app.utils.constants import LOCATIONS, AREA_CODES, SEOUL_DISTRICT_CODES, JOB_SYNONYMS

logger = logging.getLogger(__name__)

###############################################################################
# AgentState
###############################################################################
class AgentState(Dict):
    query: str
    context: List[Document]
    answer: str
    should_rewrite: bool
    rewrite_count: int
    answers: List[str]
    user_profile: Dict[str, str]  # 예: {"age":"", "location":"", "jobType":""}


###############################################################################
# JobAdvisorAgent
###############################################################################
class JobAdvisorAgent:
    """채용정보 검색 및 추천을 담당하는 에이전트"""
    
    def __init__(self, llm: ChatOpenAI, vector_search: VectorStoreSearch):
        self.llm = llm
        self.vector_search = vector_search
        self.document_filter = DocumentFilter()

    async def search_jobs(self, query: str, user_profile: Dict = None) -> Dict:
        """채용정보 검색 실행"""
        try:
            # 1. 검색을 위한 정보 추출
            location, job_type = self._extract_search_info(query, user_profile)
            logger.info(f"[JobAdvisor] NER 추출 결과 - 지역: {location}, 직무: {job_type}")
            
            # 2. 벡터 검색 실행
            search_params = {
                "지역": location,
                "직무": job_type
            }
            logger.info(f"[JobAdvisor] 검색 파라미터: {search_params}")
            
            search_results = self.vector_search.search_jobs(search_params)
            logger.info(f"[JobAdvisor] 검색 결과 수: {len(search_results) if search_results else 0}")
            
            # 3. 검색 결과가 없는 경우
            if not search_results:
                return {
                    "message": "죄송합니다. 현재 조건에 맞는 채용정보를 찾지 못했습니다.",
                    "type": "job",
                    "jobPostings": []
                }

            # 4. 검색 결과 처리
            job_postings = self._process_search_results(search_results)
            logger.info(f"[JobAdvisor] 처리된 채용공고 수: {len(job_postings)}")
            
            # 5. 최종 응답 생성
            response = {
                "message": json.dumps(job_postings[:5], ensure_ascii=False),
                "type": "job",
                "jobPostings": job_postings[:5],  # 상위 5개만 반환
                "final_answer": self._generate_response_message(query, job_postings, location, job_type)
            }
            logger.info(f"[JobAdvisor] 최종 응답 생성 완료: {response}")
            
            return response

        except Exception as e:
            logger.error(f"[JobAdvisor] 검색 중 오류: {str(e)}", exc_info=True)
            return {
                "message": f"채용정보 검색 중 오류 발생: {str(e)}",
                "type": "error",
                "jobPostings": []
            }

    def _extract_search_info(self, query: str, user_profile: Dict = None) -> tuple[str, str]:
        """검색에 필요한 지역과 직무 정보 추출"""
        try:
            # 1. 지역 정보 추출
            location = ""
            for loc in LOCATIONS:
                if loc in query:
                    location = loc
                    # 서울인 경우 구 정보도 확인
                    if loc == "서울":
                        for district in SEOUL_DISTRICT_CODES:
                            if district in query:
                                location = f"{loc} {district}"
                                break
                    break
            
            # 2. 직무 정보 추출
            job_type = ""
            # 직무 키워드 매칭
            job_keywords = JOB_SYNONYMS.keys()
            for keyword in job_keywords:
                if keyword in query:
                    job_type = keyword
                    break
            
            # 3. 사용자 프로필에서 보완
            if user_profile:
                if not location and user_profile.get("location"):
                    location = user_profile["location"]
                if not job_type and user_profile.get("job_type"):
                    job_type = user_profile["job_type"]
            
            return location, job_type
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 정보 추출 중 오류: {str(e)}")
            return "", ""

    def _process_search_results(self, documents: List[Document]) -> List[Dict]:
        """검색 결과를 JobPosting 형식으로 변환"""
        try:
            job_postings = []
            for doc in documents:
                try:
                    posting = {
                        "id": doc.metadata.get("채용공고ID", ""),
                        "title": doc.metadata.get("채용제목", ""),
                        "company": doc.metadata.get("회사명", ""),
                        "location": doc.metadata.get("근무지역", ""),
                        "salary": doc.metadata.get("급여조건", ""),
                        "workingHours": doc.metadata.get("근무시간", ""),
                        "description": doc.page_content,
                        "phoneNumber": doc.metadata.get("전화번호", ""),
                        "deadline": doc.metadata.get("마감일자", ""),
                        "requiredDocs": doc.metadata.get("제출서류", ""),
                        "hiringProcess": doc.metadata.get("전형방법", ""),
                        "insurance": doc.metadata.get("4대보험", ""),
                        "jobCategory": doc.metadata.get("직종", ""),
                        "jobKeywords": doc.metadata.get("직무키워드", ""),
                        "posting_url": doc.metadata.get("채용공고URL", "")
                    }
                    job_postings.append(posting)
                except Exception as e:
                    logger.error(f"[JobAdvisor] 결과 처리 중 오류: {str(e)}")
                    continue
                    
            return job_postings
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 결과 처리 중 오류: {str(e)}")
            return []

    def _filter_by_user_profile(self, job_postings: List[Dict], user_profile: Dict) -> List[Dict]:
        """사용자 프로필 기반 필터링"""
        try:
            if not user_profile:
                return job_postings
                
            filtered_postings = []
            for posting in job_postings:
                # 지역 매칭
                if user_profile.get("location"):
                    if user_profile["location"] not in posting["location"]:
                        continue
                        
                # 직무 매칭
                if user_profile.get("job_type"):
                    if user_profile["job_type"] not in posting["jobCategory"]:
                        continue
                        
                filtered_postings.append(posting)
                
            return filtered_postings
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 필터링 중 오류: {str(e)}")
            return job_postings

    def _generate_response_message(self, query: str, job_postings: List[Dict], location: str, job_type: str) -> str:
        """응답 메시지 생성"""
        try:
            count = len(job_postings)
            if count == 0:
                return "죄송합니다. 조건에 맞는 채용정보를 찾지 못했습니다."
                
            message_parts = []
            if location:
                message_parts.append(f"{location}지역")
            if job_type:
                message_parts.append(f"'{job_type}' 직종")
                
            location_job = " ".join(message_parts)
            if location_job:
                return f"{location_job}에서 {count}개의 채용정보를 찾았습니다."
            else:
                return f"'{query}' 검색 결과 {count}개의 채용정보를 찾았습니다."
                
        except Exception as e:
            logger.error(f"[JobAdvisor] 메시지 생성 중 오류: {str(e)}")
            return "채용정보를 찾았습니다."

    ###############################################################################
    # (A) NER 추출용 함수
    ###############################################################################
    def get_user_ner_runnable(self) -> Runnable:
        """
        사용자 입력 예: "서울 요양보호사"
        -> LLM이 아래와 같이 JSON으로 추출:
           {"직무": "요양보호사", "지역": "서울", "연령대": ""}
        """
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4o-mini",
            temperature=0.0
        )

        return EXTRACT_INFO_PROMPT | llm


    def _extract_user_ner(self, user_message: str, user_profile: Dict[str, str], chat_history: str = "") -> Dict[str, str]:
        """사용자 메시지에서 정보 추출"""
        try:
            # 1. 먼저 사전을 이용한 직접 매칭 시도
            extracted_info = {"지역": "", "직무": "", "연령대": ""}
            
            # 지역 매칭 - constants.py의 LOCATIONS 사용
            for location in LOCATIONS:
                if location in user_message:
                    extracted_info["지역"] = location
                    break
            
            # 직무 매칭 및 유의어 확인
            words = user_message.split()
            for word in words:
                if word in JOB_SYNONYMS:
                    extracted_info["직무"] = word
                    break
                # 유의어 검색
                for job, synonyms in JOB_SYNONYMS.items():
                    if word in synonyms:
                        extracted_info["직무"] = job
                        break
            
            # 2. 사전 매칭으로 충분한 정보를 얻지 못한 경우 LLM 사용
            if not (extracted_info["지역"] or extracted_info["직무"]):
                chain = EXTRACT_INFO_PROMPT | self.llm | StrOutputParser()
                response = chain.invoke({
                    "user_query": user_message,
                    "chat_history": chat_history if chat_history else ""
                })
                
                # JSON 파싱
                cleaned = response.replace("```json", "").replace("```", "").strip()
                llm_extracted = json.loads(cleaned)
                
                # LLM 결과와 사전 매칭 결과 병합
                if not extracted_info["지역"] and llm_extracted.get("지역"):
                    extracted_info["지역"] = llm_extracted["지역"]
                if not extracted_info["직무"] and llm_extracted.get("직무"):
                    extracted_info["직무"] = llm_extracted["직무"]
                if llm_extracted.get("연령대"):
                    extracted_info["연령대"] = llm_extracted["연령대"]
            
            # 3. 프로필 정보로 보완
            if user_profile:
                if not extracted_info["지역"] and user_profile.get("location"):
                    extracted_info["지역"] = user_profile["location"]
                if not extracted_info["직무"] and user_profile.get("jobType"):
                    extracted_info["직무"] = user_profile["jobType"]
                    
            logger.info(f"[JobAdvisor] NER 추출 결과: {extracted_info}")
            return extracted_info
            
        except Exception as e:
            logger.error(f"[JobAdvisor] NER 추출 중 에러: {str(e)}")
            return {"지역": "", "직무": "", "연령대": ""}


    ###############################################################################
    # (B) 일반 대화/채용정보 검색 라우팅
    ###############################################################################
    def retrieve(self, state: AgentState):
        query = state['query']
        logger.info(f"[JobAdvisor] retrieve 시작 - 쿼리: {query}")

        # (1) 일반 대화 체크
        if not self.is_job_related(query):
            # 일상 대화 처리 -> LLM으로 전달 -> 구직 관련 대화 유도
            response = self.chat_agent.chat(query)
            return {
                # ChatResponse 호환 형태
                "message": response,  # answer
                "jobPostings": [],
                "type": "info",
                "user_profile": state.get("user_profile", {}),
                "context": [],
                "query": query
            }

        # (2) job 검색
        logger.info("[JobAdvisor] 채용정보 검색 시작")
        user_profile = state.get("user_profile", {})
        user_ner = self._extract_user_ner(query, user_profile)

        try:
            results = self.vector_search.search_jobs(user_ner=user_ner, top_k=10)
            logger.info(f"[JobAdvisor] 검색 결과 수: {len(results)}")
        except Exception as e:
            logger.error(f"[JobAdvisor] 검색 중 에러 발생: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 검색 중 오류가 발생했습니다.",
                "jobPostings": [],
                "type": "info",
                "user_profile": user_profile,
                "context": [],
                "query": query
            }

        # (3) 최대 5건만 추출
        top_docs = results[:2]

        # (4) Document -> JobPosting 변환
        job_postings = []
        for i, doc in enumerate(top_docs, start=1):
            md = doc.metadata
            job_postings.append({
                "id": md.get("채용공고ID", f"no_id_{i}"),
                "location": md.get("근무지역", "위치 정보 없음"),
                "company": md.get("회사명", "회사명 없음"),
                "title": md.get("채용제목", "제목 없음"),
                "salary": md.get("급여조건", "급여 정보 없음"),
                "workingHours": md.get("근무시간", "근무시간 정보 없음"),
                "description": md.get("상세정보", doc.page_content[:500]) or "상세내용 정보 없음",
                "phoneNumber": md.get("전화번호", "전화번호 정보 없음"),
                "deadline": md.get("접수마감일", "마감일 정보 없음"),
                "requiredDocs": md.get("제출서류", "제출서류 정보 없음"),
                "hiringProcess": md.get("전형방법", "전형방법 정보 없음"),
                "insurance": md.get("사회보험", "사회보험 정보 없음"),
                "jobCategory": md.get("모집직종", "모집직종 정보 없음"),
                "jobKeywords": md.get("직종키워드", "직종키워드 정보 없음"),
                "posting_url": md.get("채용공고URL", "채용공고URL 정보 없음"),
                "rank": i
            })

        # (5) 메시지 / 타입
        if job_postings:
            msg = f"'{query}' 검색 결과, 상위 {len(job_postings)}건을 반환합니다."
            res_type = "jobPosting"
        else:
            msg = "조건에 맞는 채용공고를 찾지 못했습니다."
            res_type = "info"

        # (6) ChatResponse 호환 dict
        return {
            "message": msg,
            "jobPostings": job_postings,
            "type": res_type,
            "user_profile": user_profile,
            "context": results,  # 다음 노드(verify 등)에서 사용
            "query": query
        }

    ###############################################################################
    # (C) 이하 verify, rewrite, generate 등은 기존 로직 그대로
    ###############################################################################
    def verify(self, state: AgentState) -> dict:
        if state.get('is_greeting', False):
            return {
                "should_rewrite": False,
                "rewrite_count": 0,
                "answers": state.get('answers', [])
            }
            
        context = state['context']
        query = state['query']
        rewrite_count = state.get('rewrite_count', 0)
        
        if rewrite_count >= 3:
            return {
                "should_rewrite": False,
                "rewrite_count": rewrite_count
            }
        
        verify_chain = verify_prompt | self.llm | StrOutputParser()
        response = verify_chain.invoke({
            "query": query,
            "context": "\n\n".join([str(doc) for doc in context])
        })
        
        return {
            "should_rewrite": "NO" in response.upper(),
            "rewrite_count": rewrite_count + 1,
            "answers": state.get('answers', [])
        }

    def rewrite(self, state: AgentState) -> dict:
        """쿼리 재작성"""
        if state.get('is_greeting', False):
            return {"answer": state['query']}
            
        try:
            # prompts.py의 rewrite_prompt 사용
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            answer = rewrite_chain.invoke({
                "original_query": state['query'],
                "transformed_query": state['query']  # 변환된 쿼리
            })
            return {"answer": answer.strip()}
        except Exception as e:
            logger.error(f"[JobAdvisor] 쿼리 재작성 중 오류: {str(e)}")
            return {"answer": state['query']}

    def generate(self, state: AgentState) -> dict:
        """응답 생성"""
        try:
            # prompts.py의 generate_prompt 사용
            generate_chain = generate_prompt | self.llm | StrOutputParser()
            answer = generate_chain.invoke({
                "question": state['query'],
                "context": "\n".join([doc.page_content for doc in state['context']])
            })
            return {"answer": answer.strip()}
        except Exception as e:
            logger.error(f"[JobAdvisor] 응답 생성 중 오류: {str(e)}")
            return {"answer": "죄송합니다. 응답을 생성하는 중에 문제가 발생했습니다."}

    def router(self, state: AgentState) -> str:
        # 기본 대화나 인사인 경우 바로 generate로
        if state.get('is_basic_question', False):
            return "generate"
            
        # 검색 결과가 있으면 verify로
        if state.get('context', []):
            return "verify"
            
        # 검색 결과가 없으면 generate로
        return "generate"

    def setup_workflow(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("verify", self.verify)
        workflow.add_node("rewrite", self.rewrite)
        workflow.add_node("generate", self.generate)
        
        workflow.add_edge("retrieve", "verify")
        workflow.add_conditional_edges("verify", self.router)
        workflow.add_conditional_edges("rewrite", self.router)
        workflow.add_edge("generate", END)
        
        workflow.set_entry_point("retrieve")
        
        return workflow.compile()


    async def chat(self, query: str, user_profile: Dict = None, chat_history: List[Dict] = None) -> Dict:
        """사용자 메시지에 대한 응답을 생성합니다."""
        logger.info("=" * 50)
        logger.info("[JobAdvisor] chat 메서드 시작")
        logger.info(f"[JobAdvisor] 입력 쿼리: {query}")
        logger.info(f"[JobAdvisor] 사용자 프로필: {user_profile}")
        logger.info(f"[JobAdvisor] 대화 이력 수: {len(chat_history) if chat_history else 0}")
        logger.info(f"[JobAdvisor] 대화 이력 타입: {type(chat_history)}")
        if chat_history:
            logger.info(f"[JobAdvisor] 대화 이력 첫 번째 항목: {chat_history[0] if chat_history else None}")

        try:
            # 1. 채용정보 검색 실행
            search_result = await self.search_jobs(query, user_profile)
            
            # 2. 검색 결과가 있는 경우
            if search_result.get("jobPostings"):
                return search_result
            
            # 3. 검색 결과가 없는 경우
            return {
                "message": "죄송합니다. 현재 조건에 맞는 채용정보를 찾지 못했습니다.",
                "type": "jobPosting",
                "jobPostings": []
            }
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 채팅 처리 중 오류: {str(e)}", exc_info=True)
            return {
                "message": f"채용정보 검색 중 오류가 발생했습니다: {str(e)}",
                "type": "error",
                "jobPostings": []
            }

    def _extract_location(self, query: str) -> Tuple[str, str]:
        """쿼리에서 지역 정보를 추출합니다."""
        try:
            # 시/도 추출
            for location in LOCATIONS:
                if location in query:
                    # 서울인 경우 구 정보도 확인
                    if location == "서울":
                        for district in SEOUL_DISTRICT_CODES.keys():
                            if district in query:
                                return location, district
                    return location, ""
            return "", ""
        except Exception as e:
            logger.error(f"[JobAdvisor] 지역 정보 추출 중 에러: {str(e)}")
            return "", ""