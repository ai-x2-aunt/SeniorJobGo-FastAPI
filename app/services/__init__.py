"""
벡터 스토어 초기화 및 검색 객체 초기화
"""

import logging
from fastapi import FastAPI
from .vector_store_ingest import VectorStoreIngest
from .vector_store_search import VectorStoreSearch

logger = logging.getLogger(__name__)

def initialize_vector_store(app: FastAPI):
    """
    벡터 스토어 초기화 및 검색 객체 초기화
    """

    # 벡터 스토어 초기화
    ingest = VectorStoreIngest()  # DB 생성/로드 담당
    collection = ingest.setup_vector_store()  # Chroma 객체

    vector_search = VectorStoreSearch(collection)
    app.state.vector_search = vector_search  # 앱 상태에 저장