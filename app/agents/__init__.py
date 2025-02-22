"""
Agent 초기화 및 상태 저장 모듈
"""

from langchain_openai import ChatOpenAI
from app.agents.job_advisor import JobAdvisorAgent
from app.agents.training_advisor import TrainingAdvisorAgent
from fastapi import FastAPI

def initialize_agents(app: FastAPI):
    """
    LLM과 에이전트 초기화 및 상태 저장 함수
    """
    # LLM 초기화
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
        request_timeout=30
    )
    app.state.llm = llm  # 앱 상태에 저장
    
    # JobAdvisor 에이전트 초기화
    app.state.job_advisor_agent = JobAdvisorAgent(
        llm=llm,
        vector_search=app.state.vector_search
    )

    # TrainingAdvisor 에이전트 초기화
    app.state.training_advisor_agent = TrainingAdvisorAgent(llm)