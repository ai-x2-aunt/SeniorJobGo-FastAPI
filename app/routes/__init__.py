"""
라우터 등록 함수
"""

from fastapi import FastAPI

def register_routes(app: FastAPI):
    """
    라우터 등록 함수
    레거시 코드 라우터는 주석 처리하여 제외
    """
    try:
        from .chat_router import router as chat_router
        from .userInform_router import router as userInform_router
        from .training_router import router as training_router
        # from .regacy_router import router as regacy_router

        app.include_router(chat_router, prefix="/api/v1")
        app.include_router(userInform_router)
        app.include_router(training_router)
        # app.include_router(regacy_router)
    except Exception as e:
        raise Exception(f"라우터 등록 중 오류 발생: {str(e)}")
