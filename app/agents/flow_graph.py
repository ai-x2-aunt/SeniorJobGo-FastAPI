# app/graph/flow_graph.py
from langgraph.graph import StateGraph, START, END
from app.models.flow_state import FlowState
from app.agents.node import process_tool_output_node
from langchain_openai import ChatOpenAI
import logging
from app.agents.supervisor_node import supervisor_node

logger = logging.getLogger(__name__)

async def build_flow_graph(llm: ChatOpenAI = None) -> StateGraph:
    """상태 관리 그래프 구성"""
    
    # 그래프 초기화
    workflow = StateGraph(FlowState)
    
    # 노드 연결 - async 함수 지정
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("process_output", process_tool_output_node)
    
    # 엣지 연결
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "process_output")
    workflow.add_edge("process_output", END)
    
    return workflow.compile()