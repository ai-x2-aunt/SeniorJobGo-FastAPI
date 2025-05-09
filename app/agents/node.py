import logging
import json
from app.models.flow_state import FlowState
logger = logging.getLogger(__name__)

def process_tool_output_node(state: FlowState):
    """Tool 실행 결과를 처리하는 노드"""
    try:
        # 이미 supervisor_node에서 state.final_response가 설정되어 있음
        if not state.final_response:
            raise ValueError("final_response가 없음")
            
        # 로깅
        logger.info(f"[ProcessToolOutput] 최종 응답: {state.final_response}")
        
        # ChatResponse 형식에 맞게 응답 구성
        response = {
            "message": state.final_response.get("message", ""),
            "type": state.final_response.get("type", "chat"),
            "jobPostings": state.jobPostings,
            "trainingCourses": state.trainingCourses,
            "policyPostings": state.policyPostings,
            "mealPostings": state.mealPostings,
            "user_profile": state.user_profile,
        }
        
        # state 업데이트
        state.final_response = response
        return state
    except TimeoutError:
        logger.warning("[ProcessToolOutput] 타임아웃 발생, 재시도")
        raise
        
    except Exception as e:
        logger.error(f"[ProcessToolOutput] 오류: {str(e)}", exc_info=True)
        state.error_message = str(e)
        state.final_response = {
            "message": f"처리 중 오류 발생: {str(e)}",
            "type": "error",
            "jobPostings": [],
            "trainingCourses": [],
            "policyPostings": [],
            "mealPostings": []
        }
        return state
