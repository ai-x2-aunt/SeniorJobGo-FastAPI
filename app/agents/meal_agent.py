import time
from langchain_openai import ChatOpenAI
from cachetools import TTLCache  # 제거
from app.services.meal_data_client import PublicDataClient
import logging
from typing import Dict
from kiwipiepy import Kiwi

logger = logging.getLogger(__name__)

class MealAgent: 
  """
  무료급식소 정보를 제공하는 AI 기반 에이전트.
  사용자 질문을 분석하여 지역 정보를 추출하고, 필터링된 무료급식소 데이터를 반환합니다.
  """
  def __init__(self, data_client: PublicDataClient, llm: ChatOpenAI):
    self.data_client = data_client              # 공공데이터 API에서 데이터를 가져오는 객체
    self.llm = llm
    self.cache = TTLCache(maxsize=100, ttl=300) # 최대 100개 저장, 5분간 유지
    self.kiwi = Kiwi()  # Kiwi 객체 초기화
    self.SYSTEM_PROMPT = """
      당신은 고령층을 위한 무료 급식소 안내 전문가입니다.
      사용자가 요청한 지역(예: {region})에 해당하는 무료급식소 정보를 아래 데이터에서 찾아주세요.
      - 65세 이상 대상자 우선 제공
      - 주소 설명시 주변 지점 설명
      - 자세한 급식소 안내를 위해 사용자가 이용하고자하는 지역을 추출할 것
      - 이모티콘을 붙여 친근하고, 존댓말을 사용할 것

      데이터 형식:
      - 시설명: {fcltyNm}
      - 주소: {rdnmadr}

      질문: {query}
      """
  
  async def query_meal_agent(self, query: str) -> Dict:
    """사용자의 질문에 대해 무료급식소 정보를 제공하는 응답을 생성합니다"""
    try:
      logger.info(f"[MealAgent] 새로운 채팅 요청: {query}")
      
      start_time = time.time() # 시작 시간 기록
      print(f"DEBUG: 시작 - 사용자 입력 쿼리: {query}")
      
      # 1. 다중 지역명 추출 (예: ["성동구", "강남구"])
      regions = self._extract_region(query)
      print(f"DEBUG: 결과 - Extracted regions: '{regions}'")
      
      if not regions:
        elapsed_time = time.time() - start_time
        print(f"DEBUG: 처리 시간: {elapsed_time:.2f} 초")
        return "죄송합니다. 어느 지역의 무료 급식소를 찾으시는지 명확히 말씀해 주시겠어요?"

      # 지역명 추출
      regions = self._extract_region(query)
      if not regions:
        return {
          "message": "어느 지역의 무료 급식소를 찾으시는지 명확히 말씀해 주시겠어요? 🤔",
          "type": "meal"
        }

      # 2. 각 지역별로 캐시에서 결과 조회 (캐시 키는 단일 지역명)
      results = {}
      for region in regions:
        if region in self.cache:
          print(f"DEBUG: 캐시에 '{region}' 결과 있음")
          results[region] = self.cache[region]
        else:
          print(f"DEBUG: 캐시에 '{region}' 결과 없음")
          results[region] = None

      print(f"DEBUG: 캐시 결과 - {results}")

      # 3. 캐시 미스가 있는 경우, 전체 데이터를 불러와 각 지역에 대해 새로운 결과 생성
      if any(v is None for v in results.values()):
        all_data = self.data_client.fetch_meal_services()
        meal_postings = []

        for region in regions:
            filtered_data = self.data_client.filter_by_region(all_data, region)
            print(f"DEBUG: {region} 필터된 데이터: {filtered_data}")
            
            # 필터링된 데이터를 MealPosting 형식으로 변환
            for item in filtered_data[:5]:  # 각 지역당 최대 5개
                meal_postings.append({
                    "name": item.get("fcltyNm", ""),
                    "address": item.get("rdnmadr", ""),
                    "phone": item.get("phoneNumber", ""),
                    "operatingHours": item.get("mlsvTime", ""),
                    "targetGroup": item.get("mlsvTrget", ""),
                    "description": item.get("mlsvDate", ""),
                    "latitude": item.get("latitude", 0.0),  # 위도 추가
                    "longitude": item.get("longitude", 0.0)  # 경도 추가
                })

        # 응답 메시지 생성
        if meal_postings:
            message = f"{', '.join(regions)}에서 {len(meal_postings)}개의 무료 급식소를 찾았습니다 🍚"
        else:
            message = f"죄송합니다. 현재 {', '.join(regions)}의 무료 급식소 정보를 찾을 수 없습니다 🙏"

        return {
            "message": message,
            "type": "meal",
            "mealPostings": meal_postings
        }

    except Exception as e:
        logger.error(f"[MealAgent] 처리 중 오류: {str(e)}", exc_info=True)
        return {
            "message": f"무료급식소 정보 검색 중 오류가 발생했습니다: {str(e)}",
            "type": "error",
            "mealPostings": []
        }

  def _extract_region(self, query: str) -> list:
    """
    Kiwi를 사용하여 질문에서 지역명을 추출합니다.

    Args:
      query (str): 사용자의 질문
          
    Returns:
      list: 추출된 지역명 리스트 (예: ["성동구", "강남구"])
    """

    # Kiwi를 사용하여 텍스트 분석
    result = self.kiwi.analyze(query)
    candidates = []

    # 형태소 분석 결과 처리
    for token in result[0][0]:  # result[0][0]에 형태소 분석 결과가 있음
      morpheme = token[0]       # 형태소
      pos = token[1]            # 품사 태그
      
      # 일반 명사(NNG)나 고유 명사(NNP) 중 행정구역 접미어로 끝나는 단어 추출
      if pos in ['NNG', 'NNP'] and morpheme.endswith(('구', '시', '군', '도')):
        candidates.append(morpheme)

    # fallback: 만약 추출된 후보가 없다면, 모든 명사(NNG, NNP) 중 첫 번째 단어를 후보로 추가
    if not candidates:
      for token in result[0][0]:
        morpheme = token[0]
        pos = token[1]
        if pos in ['NNG', 'NNP']:
          candidates.append(morpheme)
          break

    # 중복 제거 (순서 유지)
    return list(dict.fromkeys(candidates))