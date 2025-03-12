from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
import logging
import os
import re
from dotenv import load_dotenv
from functools import partial
from typing import Dict, List, Union
import asyncio
from sentence_transformers import SentenceTransformer, util
from app.services.policy_data_client import Gov24Client

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('policy_agent.log')
    ]
)
logger = logging.getLogger('PolicyAgent')

# 환경변수 로드
load_dotenv()

# OpenAI 및 검색 도구 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4
)

search = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=10,
    search_depth="advanced",
    include_raw_content=True,
)

gov24_client = Gov24Client()

# 검색 도구 정의
tools = [
    Tool(
        name="Web_Search",
        description="웹에서 중장년층 관련 정보와 뉴스를 검색하여 최신 동향을 제공합니다.",
        func=partial(search.run)
    ),
    Tool(
        name="Gov24_Search",
        description="Gov24 API를 활용하여 노인 및 고령자 대상 지원 정책 정보를 조회합니다.",
        func=gov24_client.fetch_policies
    )
]

# 에이전트 생성
agent = create_react_agent(
    llm,
    tools,
    PromptTemplate.from_template(
        """
        고령자 전문 상담 에이전트입니다.

        사용 가능한 도구들:
        {tools}

        도구 이름들:
        {tool_names}

        다음 원칙을 따라주세요:
        1. 정보를 종합하여 명확하게 설명
        2. 항상 공식 URL이나 출처 제공
        3. 이해하기 쉽게 한국어로 응답
        4. 같은 검색을 반복하지 마세요
        5. 최대한 많은 정보를 찾아주세요

        질문: {input}

        {agent_scratchpad}
        """
    )
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_iterations=1,
    max_execution_time=100
)

# 텍스트 전처리
def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

# 한국어 임베딩 모델
model = SentenceTransformer('jhgan/ko-sbert-nli')

async def query_policy_agent(query: Union[str, Dict]) -> Dict:
    try:
        # 검색어 전처리
        if isinstance(query, dict):
            search_query = query.get('query', '')
            user_ner = query.get('user_ner', {})
            if user_ner.get('연령대'):
                search_query = f"{user_ner['연령대']} {search_query}"
        else:
            search_query = str(query)
            
        logger.info(f"[PolicyAgent] 정책 검색 시작: {search_query}")
        
        # Gov24 API 검색
        try:
            gov24_policies = gov24_client.fetch_policies(search_query)
            logger.info(f"[PolicyAgent] Gov24 검색 결과 수: {len(gov24_policies)}")
        except Exception as e:
            logger.error(f"[PolicyAgent] Gov24 API 오류: {str(e)}")
            gov24_policies = []

        # URL 중복 체크 및 결과 저장
        seen_urls = set()
        gov24_filtered = []
        web_filtered = []

        # Gov24 결과 처리
        if gov24_policies:
            query_embedding = model.encode(search_query, convert_to_tensor=True)
            gov24_scored = []
            
            for policy in gov24_policies:
                url = policy.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    policy_text = f"{policy.get('title', '')} {policy.get('content', '')} {policy.get('target', '')}"
                    policy_embedding = model.encode(policy_text, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(query_embedding, policy_embedding)[0][0].item()
                    gov24_scored.append((policy, similarity))
            
            gov24_sorted = sorted(gov24_scored, key=lambda x: x[1], reverse=True)
            gov24_filtered = [item[0] for item in gov24_sorted[:3]]
        
        # 웹 검색 실행
        try:
            async with asyncio.timeout(100):
                web_query = f"고령자 대상 {search_query}"
                logger.info(f"[Web Search] 검색 시작 - 쿼리: {web_query}")
                
                web_results = await search.arun(web_query)
                logger.info(f"[Web Search] Raw API 응답: {web_results}")
                
                if isinstance(web_results, list):
                    query_embedding = model.encode(search_query, convert_to_tensor=True)
                    web_scored = []

                    for item in web_results:
                        try:
                            title = item.get('title', '')
                            content = item.get('content', '') or item.get('snippet', '')
                            url = item.get('url', '') or item.get('link', '')
                            
                            doc_text = f"{title} {content}"[:1000]
                            doc_embedding = model.encode(doc_text, convert_to_tensor=True)
                            similarity = util.pytorch_cos_sim(query_embedding, doc_embedding)[0][0].item()
                            
                            domain = url.split('/')[2].replace('www.', '') if url else "정보 없음"
                            content_summary = content[:300] + "..." if content and len(content) > 300 else content
                            content_summary = clean_text(content_summary) if content_summary else "정보 없음"
                            
                            policy_info = {
                                "source": domain,
                                "title": title.strip() if title else "정보 없음",
                                "content": content_summary,
                                "target": "고령층",
                                "applyMethod": item.get('applyMethod', "정보 없음"),
                                "applicationPeriod": item.get('applicationPeriod', "정보 없음"),
                                "supplytype": item.get('supplytype', "정보 없음"),
                                "contact": item.get('contact', "정보 없음"),
                                "url": url if url else "정보 없음"
                            }
                            web_scored.append((policy_info, similarity))
                            
                        except Exception as e:
                            logger.error(f"[Web Search] 결과 처리 오류: {str(e)}", exc_info=True)
                            continue
                    
                    web_scored.sort(key=lambda x: x[1], reverse=True)
                    web_filtered = [item[0] for item in web_scored[:2]]
                    
        except Exception as e:
            logger.error(f"[Web Search] 오류: {str(e)}")
            web_filtered = []

        # 결과 합치기
        all_policies = gov24_filtered + web_filtered
        
        if not all_policies:
            return {
                "message": "검색 결과가 없습니다. 다른 검색어로 시도해보세요.",
                "policyPostings": [],
                "type": "policy"
            }

        logger.info(f"검색어 '{search_query}'에 대해 Gov24 {len(gov24_filtered)}개, 웹 검색 {len(web_filtered)}개의 정책 찾음")
        
        return {
            "policyPostings": all_policies,
            "type": "policy"
        }

    except Exception as e:
        logger.error(f"[PolicyAgent] 오류 발생: {str(e)}")
        return {
            "message": "검색 중 오류가 발생했습니다.",
            "policyPostings": [],
            "type": "error"
        }
