import requests
from typing import Dict, List
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class Gov24Client:
    def __init__(self):
        """Gov24 API를 통해 정책 정보를 가져오는 클라이언트"""
        self.base_url = "https://api.odcloud.kr/api/gov24/v3"
        self.api_key = settings.GOV24_API_KEY

    def fetch_policies(self, query: str, page: int = 1, per_page: int = 500) -> List[Dict]:
        """Gov24 API를 사용하여 정책 검색"""
        try:
            url = f"{self.base_url}/serviceList"
            params = {
                "page": page,
                "perPage": per_page,
                "serviceKey": self.api_key,
                "searchType": "ALL",
                "searchWord": query
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get('data'):
                return []

            policies = []
            for policy in data.get('data', []):
                try:
                    # 2024년 이후 정책만 필터링
                    update_date = policy.get('수정일시', '')
                    if update_date and int(update_date[:4]) < 2025:
                        continue

                    title = policy.get('서비스명', '')
                    desc = policy.get('서비스목적요약', '')
                    target = policy.get('지원대상', '')

                    # 지원대상에 고령자 관련 키워드가 포함된 정책만 필터링
                    target_keywords = ['노인', '고령', '중장년', '어르신', '시니어', '장년', '65세']
                    if any(keyword in target for keyword in target_keywords):
                        normalized = {
                            "source": "Gov24",
                            "title": title,
                            "target": target,  # 원래 지원대상 정보 유지
                            "content": f"{title}: {desc}",
                            "applyMethod": policy.get('신청방법', '정보 없음'),
                            "applicationPeriod": policy.get('신청기한', '정보 없음'),
                            "supplytype": policy.get('지원유형', '정보 없음'),
                            "contact": policy.get('전화문의', '정보 없음'),
                            "url": policy.get('상세조회URL', '정보 없음')
                        }
                        policies.append(normalized)

                except Exception as e:
                    logger.error(f"정책 정규화 중 오류: {str(e)}")
                    continue

            return policies

        except Exception as e:
            logger.error(f"정책 검색 중 오류: {str(e)}")
            return []

    # def filter_by_region(self, data: list, region: str) -> list:
    #     """지역 정보로 정책 필터링"""
    #     region_parts = region.split()
        
    #     filtered = []
    #     for item in data:
    #         title = item.get("title", "")
    #         content = item.get("content", "")
    #         # 모든 region_parts가 제목이나 내용에 포함되어 있는지 확인
    #         if all(part in title or part in content for part in region_parts):
    #             filtered.append(item)
        
    #     return filtered 