from typing import Dict, List, Tuple
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app.core.prompts import verify_prompt, rewrite_prompt, generate_prompt, chat_prompt, EXTRACT_INFO_PROMPT, CLASSIFY_INTENT_PROMPT
from app.utils.constants import LOCATIONS, DICTIONARY
from app.agents.chat_agent import ChatAgent
from app.agents.training_advisor import TrainingAdvisorAgent
from app.services.vector_store_search import VectorStoreSearch
import logging
import os
import json

from langchain_openai import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable

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
    def __init__(self, llm, vector_search: VectorStoreSearch):
        """
        llm: OpenAI LLM
        vector_search: 다단계 검색을 수행할 VectorStoreSearch 객체
        """
        self.llm = llm
        self.vector_search = vector_search
        self.chat_agent = ChatAgent(llm)
        self.training_agent = TrainingAdvisorAgent(llm)
        self.workflow = self.setup_workflow()
        
        # 기본 프롬프트 템플릿
        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", "당신은 구직자를 돕는 전문 취업 상담사입니다."),
            ("user", "{query}")
        ])

    async def classify_intent(self, query: str, chat_history: str = "") -> Tuple[str, float]:
        """사용자 메시지의 의도를 분류합니다."""
        try:
            # 의도 분류 프롬프트 실행
            chain = CLASSIFY_INTENT_PROMPT | self.llm | StrOutputParser()
            response = await chain.ainvoke({
                "user_query": query,
                "chat_history": chat_history
            })
            
            # JSON 파싱
            try:
                result = json.loads(response)
                intent = result.get("intent", "general")
                confidence = float(result.get("confidence", 0.5))
                explanation = result.get("explanation", "")
                
                logger.info(f"[JobAdvisor] 의도 분류 결과 - 의도: {intent}, 확신도: {confidence}, 설명: {explanation}")
                
                return intent, confidence
                
            except json.JSONDecodeError as e:
                logger.error(f"[JobAdvisor] 의도 분류 결과 JSON 파싱 실패: {response}")
                logger.error(f"파싱 에러: {str(e)}")
                return "general", 0.5
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 의도 분류 중 에러: {str(e)}")
            return "general", 0.5

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
            chain = EXTRACT_INFO_PROMPT | self.llm | StrOutputParser()
            response = chain.invoke({
                "user_query": user_message,
                "chat_history": chat_history if chat_history else ""
            })
            
            # JSON 파싱
            cleaned = response.replace("```json", "").replace("```", "").strip()
            user_ner = json.loads(cleaned)
            
            # 프로필 정보로 보완
            if user_profile:
                if not user_ner.get("지역") and user_profile.get("location"):
                    user_ner["지역"] = user_profile["location"]
                if not user_ner.get("직무") and user_profile.get("jobType"):
                    user_ner["직무"] = user_profile["jobType"]
                    
            logger.info(f"[JobAdvisor] NER 추출 결과: {user_ner}")
            return user_ner
            
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
        top_docs = results[:5]

        # (4) Document -> JobPosting 변환
        job_postings = []
        for i, doc in enumerate(top_docs, start=1):
            md = doc.metadata
            job_postings.append({
                "id": md.get("채용공고ID", "no_id"),
                "location": md.get("근무지역", ""),
                "company": md.get("회사명", ""),
                "title": md.get("채용제목", ""),
                "salary": md.get("급여조건", ""),
                "workingHours": md.get("근무시간", "정보없음"),
                "description": md.get("상세정보", doc.page_content[:200]),
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
        query = state['query']
        
        rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
        response = rewrite_chain.invoke({'query': query})
        
        return {'query': response}

    def generate(self, state: AgentState) -> dict:
        query = state['query']
        context = state.get('context', [])

        # 1) jobPostings (이미 retrieve에서 만든 5건)
        job_postings = state.get("jobPostings", [])

        if not context or not job_postings:
            return {
                "message": "죄송합니다. 관련된 구인정보를 찾지 못했습니다.",
                "jobPostings": [],
                "type": "info",
                "user_profile": state.get("user_profile", {})
            }

        # 2) RAG 프롬프트
        rag_chain = generate_prompt | self.llm | StrOutputParser()
        doc_text = "\n\n".join([
            f"제목: {doc.metadata.get('채용제목', '')}\n"
            f"회사: {doc.metadata.get('회사명', '')}\n"
            f"지역: {doc.metadata.get('근무지역', '')}\n"
            f"급여: {doc.metadata.get('급여조건', '')}\n"
            f"상세내용: {doc.page_content}"
            for doc in context[:5]  # 혹은 job_postings의 길이
        ])
        response_text = rag_chain.invoke({
            "question": query,
            "context": doc_text
        })

        return {
            "message": f"최종 답변:\n{response_text}",
            "jobPostings": job_postings,  # retrieve에서 만든 것 재사용
            "type": "jobPosting",
            "user_profile": state.get("user_profile", {})
        }

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
        workflow.add_edge("verify", "rewrite")
        workflow.add_edge("verify", "generate")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("generate", END)
        
        workflow.set_entry_point("retrieve")
        
        return workflow.compile()


    async def chat(self, query: str, user_profile: Dict = None, chat_history: List[Dict] = None) -> Dict:
        """사용자 메시지에 대한 응답을 생성합니다."""
        logger.info("=" * 50)
        logger.info(f"[JobAdvisor] chat 메서드 시작")
        logger.info(f"[JobAdvisor] 입력 쿼리: {query}")
        logger.info(f"[JobAdvisor] 사용자 프로필: {user_profile}")
        logger.info(f"[JobAdvisor] 대화 이력 수: {len(chat_history) if chat_history else 0}")
        logger.info(f"[JobAdvisor] 대화 이력 타입: {type(chat_history)}")
        if chat_history:
            logger.info(f"[JobAdvisor] 대화 이력 첫 번째 항목: {chat_history[0] if chat_history else None}")

        try:
            # 대화 이력 포맷팅
            logger.info("[JobAdvisor] 대화 이력 포맷팅 시작")
            formatted_history = ""
            if chat_history and isinstance(chat_history, str):
                logger.info("[JobAdvisor] 대화 이력이 문자열 형태")
                formatted_history = chat_history
            elif chat_history and isinstance(chat_history, list):
                logger.info("[JobAdvisor] 대화 이력이 리스트 형태")
                for msg in chat_history[-5:]:  # 최근 5개 메시지만 사용
                    logger.info(f"[JobAdvisor] 메시지 처리: {msg}")
                    if isinstance(msg, dict):
                        role = "사용자" if msg.get("role") == "user" else "시스템"
                        content = msg.get("content", "")
                        if isinstance(content, dict):
                            content = content.get("message", "")
                        formatted_history += f"{role}: {content}\n"
            logger.info(f"[JobAdvisor] 포맷팅된 대화 이력: {formatted_history}")
            
            # 의도 분류 (대화 이력 포함)
            logger.info("[JobAdvisor] 의도 분류 시작")
            intent, confidence = await self.classify_intent(query, formatted_history)
            logger.info(f"[JobAdvisor] 의도 분류 결과 - 의도: {intent}, 확신도: {confidence}")
            
            # 높은 확신도의 의도에 따라 처리
            if confidence > 0.6:
                if intent == "job":
                    logger.info("[JobAdvisor] 채용정보 검색 처리 시작")
                    return await self.handle_job_query(query, user_profile, formatted_history)
                elif intent == "training":
                    logger.info("[JobAdvisor] 훈련정보 검색 처리 시작")
                    return await self.handle_training_query(query, user_profile)
            
            # 낮은 확신도 또는 일반 대화인 경우
            logger.info("[JobAdvisor] 일반 대화 처리 시작")
            return await self.handle_general_conversation(query, formatted_history)
            
        except Exception as e:
            logger.error(f"[JobAdvisor] 채팅 처리 중 에러: {str(e)}", exc_info=True)
            return {
                "message": "죄송합니다. 요청을 처리하는 중에 문제가 발생했습니다.",
                "type": "error"
            }

    async def handle_job_query(self, query: str, user_profile: Dict, chat_history: str = "") -> Dict:
        """채용정보 검색 처리"""
        logger.info("[JobAdvisor] 채용정보 검색 시작")
        try:
            # 대화 이력을 포함하여 NER 추출 (async 제거)
            user_ner = self._extract_user_ner(query, user_profile, chat_history)
            logger.info(f"[JobAdvisor] NER 추출 결과: {user_ner}")

            if not any(user_ner.values()):
                logger.info("[JobAdvisor] NER 추출 실패")
                return {
                    "message": "죄송합니다. 어떤 종류의 일자리를 찾으시는지 조금 더 자세히 말씀해 주시겠어요?",
                    "jobPostings": [],
                    "type": "info",
                    "user_profile": user_profile
                }

            results = self.vector_search.search_jobs(user_ner=user_ner, top_k=10)
            logger.info(f"[JobAdvisor] 검색 결과 수: {len(results)}")

            if not results:
                return {
                    "message": "현재 조건에 맞는 채용정보를 찾지 못했습니다. 다른 조건으로 찾아보시겠어요?",
                    "jobPostings": [],
                    "type": "info",
                    "user_profile": user_profile
                }

            top_docs = results[:5]
            job_postings = []
            
            for i, doc in enumerate(top_docs, start=1):
                try:
                    md = doc.metadata
                    job_postings.append({
                        "id": md.get("채용공고ID", f"no_id_{i}"),
                        "location": md.get("근무지역", "위치 정보 없음"),
                        "company": md.get("회사명", "회사명 없음"),
                        "title": md.get("채용제목", "제목 없음"),
                        "salary": md.get("급여조건", "급여 정보 없음"),
                        "workingHours": md.get("근무시간", "근무시간 정보 없음"),
                        "description": md.get("상세정보", doc.page_content[:300]),
                        "rank": i
                    })
                except Exception as doc_error:
                    logger.error(f"[JobAdvisor] 문서 {i} 처리 중 에러: {str(doc_error)}")
                    continue

            return {
                "message": f"'{query}' 검색 결과, {len(job_postings)}건의 채용정보를 찾았습니다.",
                "jobPostings": job_postings,
                "type": "jobPosting",
                "user_profile": user_profile
            }

        except Exception as search_error:
            logger.error("[JobAdvisor] 검색 중 에러 발생", exc_info=True)
            return {
                "message": "죄송합니다. 채용정보 검색 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "jobPostings": [],
                "type": "error",
                "user_profile": user_profile
            }

    def _deduplicate_training_courses(self, courses: List[Dict]) -> List[Dict]:
        """훈련과정 목록에서 중복된 과정을 제거합니다."""
        unique_courses = {}
        for course in courses:
            course_id = course.get('id')
            if course_id not in unique_courses:
                unique_courses[course_id] = course
        return list(unique_courses.values())

    async def handle_training_query(self, query: str, user_profile: Dict) -> Dict:
        """훈련과정 관련 질의 처리"""
        try:
            # 훈련과정 검색
            training_advisor = TrainingAdvisorAgent(self.llm)
            training_results = await training_advisor.search_training_courses(query, user_profile)
            
            # 중복 제거
            if training_results.get('trainingCourses'):
                training_results['trainingCourses'] = self._deduplicate_training_courses(training_results['trainingCourses'])
                training_results['message'] = training_results['message'].replace(
                    str(len(training_results['trainingCourses']) + 1),  # 원래 개수
                    str(len(training_results['trainingCourses']))  # 중복 제거 후 개수
                )
            
            return training_results

        except Exception as e:
            logger.error(f"[JobAdvisor] 훈련과정 검색 중 오류: {str(e)}")
            return {
                "message": "죄송합니다. 훈련과정 검색 중 오류가 발생했습니다.",
                "trainingCourses": [],
                "type": "training",
                "user_profile": user_profile
            }

    async def handle_general_conversation(self, query: str, chat_history: str = "") -> Dict:
        """일반 대화 처리"""
        logger.info("[JobAdvisor] 일반 대화로 판단")
        chat_response = await self.chat_agent.chat(query, chat_history)
        return {
            "message": chat_response,
            "jobPostings": [],
            "type": "info",
            "user_profile": {}
        }