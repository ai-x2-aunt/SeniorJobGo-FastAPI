import os
import re
import json
import logging
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid


load_dotenv(override=True)

logger = logging.getLogger(__name__)


###############################################################################
# 1) 임베딩 클래스
###############################################################################

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()  # NumPy 배열을 Python 리스트로 변환
        
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()  # NumPy 배열을 Python 리스트로 변환



###############################################################################
# 2) VectorStoreIngest: DB 생성(빌드) 및 설정
###############################################################################
class VectorStoreIngest:
    """
    - JSON 문서를 로드하여 Chroma DB에 저장(또는 갱신)
    - NER 전처리(직무, 근무 지역 등) 수행
    - setup_vectorstore()로 DB 생성 또는 로드
    """


    def __init__(self, persist_directory: Optional[str] = None):
        self.embedding_model = SentenceTransformerEmbeddings("nlpai-lab/KURE-v1")

        # 기본 경로 설정
        if persist_directory is None:
            self.persist_directory = str(Path(__file__).parent.parent.parent / "jobs_collection")
        else:
            self.persist_directory = persist_directory


        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        self._collection = None

    def setup_vector_store(self) -> Chroma:
        """
        만약 persist_directory에 Chroma DB가 존재하면 로드,
        없거나 문서가 비어 있으면 build_vectorstore("documents/jobs.json") 실행
        """
        if not os.path.exists(self.persist_directory):
            logger.info("벡터 DB가 없습니다. 새로 생성합니다.")
            self._collection = self.build_vectorstore("documents/jobs.json")
            return self._collection
        
        logger.info(f"Loading vector store from {self.persist_directory}")
        try:
            self._collection = Chroma(
                embedding_function=self.embedding_model,
                collection_name="job_postings",
                persist_directory=self.persist_directory
            )
            doc_count = self._collection._collection.count()
            logger.info(f"벡터 스토어에 저장된 문서 수: {doc_count}")
            
            if doc_count == 0:
                logger.info("벡터 DB가 비어있습니다. 새로 생성합니다.")
                self._collection = self.build_vectorstore("documents/jobs.json")
            
            return self._collection
            
        except Exception as e:
            logger.error(f"Vector store loading failed: {e}", exc_info=True)
            raise

    def build_vectorstore(self, json_file: str) -> Chroma:
        """
        jobs.json(예시)을 로드하여 문서별 NER 추출 및 청크 후,
        Chroma.from_documents()로 DB 구축
        """
        data = self._load_data(json_file)
        documents, doc_ids = self._prepare_documents(data)
            
        client_settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
        
        collection = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            ids=doc_ids,
            collection_name="job_postings",
            persist_directory=self.persist_directory,
            client_settings=client_settings
        )
        
        total_docs = collection._collection.count()
        logger.info(f"Stored {total_docs} documents in Chroma.")
        return collection

    ###############################################################################
    # 내부 메서드: 데이터 로드, NER, 청크
    ###############################################################################
    def _load_data(self, json_file: str) -> dict:
        """JSON 파일 로드"""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data.get('채용공고목록', []))} postings from {json_file}.")
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {"채용공고목록": []}

    def _clean_text(self, text: str) -> str:
        """HTML 태그, \n 등의 불필요한 문자 제거"""
        if not isinstance(text, str):
            return ""
        return re.sub(r"<[^>]+>", "", text).replace("\n", " ").strip()

    def _simplify_metadata(self, metadata: dict) -> dict:
        """복잡한 메타데이터를 Chroma에서 지원하는 단순한 형식으로 변환합니다."""
        simplified_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                # 딕셔너리나 리스트를 JSON 문자열로 변환
                simplified_metadata[key] = json.dumps(value, ensure_ascii=False)
            else:
                simplified_metadata[key] = value
        return simplified_metadata

    def _prepare_documents(self, data: dict) -> Tuple[List[Document], List[str]]:
        """jobs.json -> Document 리스트 변환, NER 수행"""
        documents = []
        doc_ids = []

        for idx_item, item in enumerate(data.get("채용공고목록", [])):
            posting_id = item.get("공고번호", "no_id")
            job_title = self._clean_text(item.get("채용제목", ""))
            company_name = self._clean_text(item.get("회사명", ""))
            work_location = self._clean_text(item.get("근무지역", ""))
            salary_condition = self._clean_text(item.get("급여조건", ""))
            job_posting_url = self._clean_text(item.get("채용공고URL", ""))

            details = item.get("상세정보", {})
            job_description = ""
            requirements_text = ""
            
            if isinstance(details, dict):
                job_description = self._clean_text(details.get("직무내용", ""))
                requirements_list = details.get("세부요건", [])
                for requirement in requirements_list:
                    for k, v in requirement.items():
                        if isinstance(v, list):
                            requirements_text += f"{k}: {' '.join(v)}\n"
                        else:
                            requirements_text += f"{k}: {v}\n"
            else:
                job_description = self._clean_text(str(details))
            
            combined_text = (
                f"{job_title}\n"
                f"회사명: {company_name}\n"
                f"근무지역: {work_location}\n"
                f"급여조건: {salary_condition}\n"
                f"{job_description}\n{requirements_text}"
            )

            # NER
            ner_data = {}
            try:
                ner_data = self._perform_ner(combined_text)
                # NER 결과 단순화
                ner_data = self._simplify_metadata(ner_data)
            except Exception as e:
                logger.warning(f"[{posting_id}] NER invoke fail: {e}")
            
            # 텍스트 청크
            splits = self.text_splitter.split_text(combined_text)
            for idx_chunk, chunk_text in enumerate(splits):
                # UUID를 사용하여 고유 ID 생성
                doc_id = str(uuid.uuid4())  # 완전히 고유한 ID 생성
                
                chunk_metadata = {
                    "채용공고ID": posting_id,
                    "채용제목": job_title,
                    "회사명": company_name,
                    "근무지역": work_location,
                    "급여조건": salary_condition,
                    "직무내용": job_description,
                    "세부요건": requirements_text,
                    "채용공고URL": job_posting_url,
                    "chunk_index": idx_chunk,
                    "unique_id": doc_id  # 메타데이터에도 UUID 저장
                }
                
                # 메타데이터 단순화
                chunk_metadata.update(ner_data)
                chunk_metadata = self._simplify_metadata(chunk_metadata)
                
                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                documents.append(doc)
                doc_ids.append(doc_id)  # UUID를 doc_ids 리스트에 추가

            logger.info(
                f"[{idx_item+1}] 공고번호: {posting_id}, "
                f"문서 {len(splits)}개 생성, NER keys={list(ner_data.keys())}"
            )

        logger.info(f"총 {len(documents)}개의 Document가 생성되었습니다.")
        return documents, doc_ids

    def _perform_ner(self, text: str) -> dict:
        """LLM 기반 NER 추출"""
        llm = ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4o-mini",
            temperature=0.0
        )

        ner_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "다음 채용공고 텍스트에서 주요 정보를 JSON 형식으로 추출해줘.\n\n"
                "추출해야 할 정보:\n"
                "- 직무\n"
                "- 회사명\n"
                "- 근무지역\n"
                "- 연령대\n"
                "- 경력조건\n"
                "- 학력\n"
                "- 고용형태\n"
                "- 모집인원\n"
                "- 장애인 채용인원\n"
                "- 모집직종\n"
                "- 직종키워드\n"
                "- 임금조건\n"
                "- 근무시간\n"
                "- 근무형태\n"
                "- 사회보험\n"
                "- 퇴직급여\n"
                "- 접수마감일\n"
                "- 전형방법\n"
                "- 접수방법\n"
                "- 제출서류\n"
                "- 전화번호\n"
                "- 휴대폰번호\n"
                "- 팩스번호\n"
                "- 이메일\n"
                "- 채용공고 등록일시\n\n"
                             
                "채용 공고:\n{text}\n\n"
                "채용 공고 내에 없는 정보는 비워두거나 적절한 방식으로 처리하고, "
                "위 정보를 JSON으로만 응답해줘."
            )
        )

        chain = ner_prompt | llm
        ner_result = chain.invoke({"text": text})
        ner_str = ner_result.content if hasattr(ner_result, "content") else str(ner_result)
        ner_str = ner_str.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(ner_str)
        except json.JSONDecodeError:
            logger.warning(f"NER JSON parse fail: {ner_str}")
            return {}
