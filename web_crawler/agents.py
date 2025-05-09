from typing import Dict, List, TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI

from .tools import CrawlerTools
from .config import SEARCH_KEYWORDS, CRAWLING_SITES

class AgentState(TypedDict):
    messages: List[str]
    current_site: str
    current_keyword: str
    crawled_data: List[Dict]
    errors: List[str]

def create_crawler_agent():
    # 크롤러 도구 초기화
    crawler_tools = CrawlerTools()
    
    # LLM 초기화
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini"
    )
    
    # 상태 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 정의
    def process_site_and_keyword(state: AgentState) -> dict:
        """사이트와 키워드 처리"""
        messages = state["messages"]
        messages.append(f"현재 처리 중: {state['current_site']} - {state['current_keyword']}")
        return {"next": "crawl_jobs", "state": state}
    
    def crawl_jobs(state: AgentState) -> dict:
        """채용 정보 크롤링"""
        try:
            jobs = crawler_tools.search_jobs.invoke(
                site=state["current_site"],
                keyword=state["current_keyword"]
            )
            state["crawled_data"].extend(jobs)
            state["messages"].append(
                f"{state['current_site']}에서 {len(jobs)}개의 채용 공고를 수집했습니다."
            )
        except Exception as e:
            error_msg = f"크롤링 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["messages"].append(error_msg)
            
        # 다음 상태 결정
        all_sites = list(CRAWLING_SITES.keys())
        current_site_index = all_sites.index(state["current_site"])
        
        if current_site_index < len(all_sites) - 1:
            # 다음 사이트로 이동
            state["current_site"] = all_sites[current_site_index + 1]
            return {"next": "process_site_keyword", "state": state}
        else:
            # 모든 사이트 처리 완료
            state["messages"].append("모든 사이트의 크롤링이 완료되었습니다.")
            return {"next": END, "state": state}
    
    # 노드 추가
    workflow.add_node("process_site_keyword", process_site_and_keyword)
    workflow.add_node("crawl_jobs", crawl_jobs)
    
    # 엣지 추가
    workflow.add_edge(START, "process_site_keyword")
    
    # 그래프 컴파일
    app = workflow.compile()
    
    def run_crawler(keywords: List[str] = SEARCH_KEYWORDS):
        """크롤러 실행"""
        for keyword in keywords:
            # 초기 상태 설정
            initial_state = AgentState(
                messages=[f"키워드 '{keyword}' 에 대한 크롤링을 시작합니다."],
                current_site=list(CRAWLING_SITES.keys())[0],
                current_keyword=keyword,
                crawled_data=[],
                errors=[]
            )
            
            # 워크플로우 실행
            final_state = app.invoke(initial_state)
            
            # 결과 출력
            for message in final_state["messages"]:
                print(message)
            
            if final_state["errors"]:
                print("\n발생한 오류:")
                for error in final_state["errors"]:
                    print(f"- {error}")
    
    return run_crawler 