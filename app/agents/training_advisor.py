from typing import Dict, List
import logging
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI

from work24.training_collector import TrainingCollector
from work24.common_codes import CommonCodeCollector, CommonCodeType
from app.services.document_filter import DocumentFilter
from app.utils.constants import AREA_CODES, INTEREST_NCS_MAPPING, AREA_ALIASES, get_job_synonyms, DISTRICT_CODES

logger = logging.getLogger(__name__)

class TrainingAdvisorAgent:
    """교육/훈련 과정 검색 및 추천을 담당하는 에이전트"""
    ###############################################################################
    # 에이전트 초기화 및 설정
    ###############################################################################
    # LLM 모델, 훈련정보 수집기, 문서 필터 초기화
    # 지역별/서울 구별 코드 매핑 데이터 설정
    # 훈련과정 검색을 위한 기반 인프라 구성
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.collector = TrainingCollector()
        self.document_filter = DocumentFilter()
        self.code_collector = CommonCodeCollector()
        
        # constants.py에서 정의된 코드 사용
        self.area_codes = AREA_CODES
        self.area_medium_codes = DISTRICT_CODES
        self.interest_ncs_mapping = INTEREST_NCS_MAPPING
        self.area_aliases = AREA_ALIASES
        
    def _initialize_area_codes(self):
        """지역 중분류 코드 초기화"""
        try:
            # 서울(11) 지역의 중분류 코드 조회
            medium_codes = self.code_collector.fetch_common_codes(
                CommonCodeType.TRAINING_AREA_MEDIUM,
                option1="11"  # 서울
            )
            
            # 중분류 코드 매핑 생성
            if isinstance(medium_codes, list):
                for code in medium_codes:
                    # XML 응답의 필드명에 맞게 수정
                    area_name = code.get("rsltCodenm", "").strip()
                    area_code = code.get("rsltCode", "")
                    if area_name and area_code:
                        # "서울" 제거 및 정규화
                        area_name = area_name.replace("서울", "").strip()
                        self.area_medium_codes[area_name] = area_code
                        # 구 없는 버전도 추가 (예: "강남구" -> "강남")
                        if area_name.endswith("구"):
                            self.area_medium_codes[area_name[:-1]] = area_code
                        
                logger.info(f"[TrainingAdvisor] 지역 중분류 코드 초기화 완료: {len(self.area_medium_codes)}개")
                logger.info(f"[TrainingAdvisor] 지역 중분류 코드: {self.area_medium_codes}")
            else:
                logger.error(f"[TrainingAdvisor] 중분류 코드 조회 실패: {medium_codes}")
            
        except Exception as e:
            logger.error(f"[TrainingAdvisor] 지역 코드 초기화 중 오류: {str(e)}")
            logger.error(f"[TrainingAdvisor] medium_codes: {medium_codes if 'medium_codes' in locals() else 'Not available'}")

    def _normalize_location(self, location: str) -> tuple[str, str]:
        """지역명을 표준화된 형식으로 변환"""
        try:
            if not location:
                return "", ""
                
            # 공백 제거 및 정규화
            location = location.strip()
            parts = location.split()
            
            city = ""
            district = ""
            
            # 시/도 처리
            if parts:
                # 첫 번째 부분이 시/도
                if parts[0] in self.area_aliases:
                    city = self.area_aliases[parts[0]]
                    
                    # 구/군 정보 처리
                    if len(parts) > 1:
                        district_name = parts[1]
                        
                        # "구" 접미사 처리
                        if not district_name.endswith("구") and not district_name.endswith("군"):
                            if city == "서울":
                                district_name += "구"
                            elif city == "경기":
                                if district_name in ["연천", "가평", "양평"]:
                                    district_name += "군"
                                else:
                                    district_name += "시"
                        
                        # 시군구 코드 매핑에서 찾기
                        district_key = f"{district_name}"
                        if district_key in DISTRICT_CODES:
                            district = district_name
                            
                            # 코드 검증 - 상위 시도 코드와 일치하는지
                            district_code = DISTRICT_CODES[district_key]
                            city_code = AREA_CODES[city]
                            if not district_code.startswith(city_code[:2]):
                                logger.warning(f"[TrainingAdvisor] 시도-시군구 코드 불일치: city={city}({city_code}), district={district}({district_code})")
                                district = ""
            
            logger.info(f"[TrainingAdvisor] 지역명 정규화: {location} -> city={city}, district={district}")
            return city, district
            
        except Exception as e:
            logger.error(f"[TrainingAdvisor] 지역명 정규화 중 오류: {str(e)}")
            return "", ""

    ###############################################################################
    # 검색 파라미터 구성
    ###############################################################################
    # 추출된 개체명 기반 검색 조건 구성
    # 지역 코드 변환(시/도 -> 코드, 서울 구별 코드 처리)
    # 직무 키워드 추출 및 검색어 생성
    # 기본 날짜 범위 설정(오늘 ~ 3개월 후)
    # 검색 파라미터 반환
    def _build_search_params(self, user_ner: Dict, user_profile: Dict = None) -> Dict:
        """검색 파라미터 구성"""
        try:
            logger.info(f"[TrainingAdvisor] 검색 파라미터 구성 시작")
            logger.info(f"[TrainingAdvisor] 사용자 NER: {user_ner}")
            logger.info(f"[TrainingAdvisor] 사용자 프로필: {user_profile}")
            
            # 오늘부터 6개월 내 시작하는 과정 검색
            today = datetime.now()
            half_year_later = today + timedelta(days=180)
            
            # Work24 API 필수 파라미터
            params = {
                "returnType": "JSON",                  # 필수: 리턴타입
                "outType": "1",                        # 필수: 출력형태 (1:리스트)
                "pageNum": "1",                        # 필수: 시작페이지
                "pageSize": "100",                      # 필수: 페이지당 출력건수
                "srchTraStDt": today.strftime("%Y%m%d"),  # 오늘부터
                "srchTraEndDt": half_year_later.strftime("%Y%m%d"),  # 6개월 후까지
                "sort": "ASC",                        # 필수: 정렬방법
                "sortCol": "TRNG_BGDE",               # 필수: 정렬컬럼
                
                # 검색 조건
                "srchTraArea1": "",                    # 훈련지역 대분류
                "srchTraArea2": "",                    # 훈련지역 중분류
                "srchNcs1": "",                        # NCS 대분류
                "srchNcs2": "",                        # NCS 중분류
                "srchNcs3": "",                        # NCS 소분류
                "crseTracseSe": "C0061",              # 훈련과정 구분 (국민내일배움카드)
                "srchTraGbn": "",                      # 훈련구분
                "srchTraType": "",                     # 훈련유형
                "srchTraProcessNm": "",                # 훈련과정명
                "srchTraOrganNm": ""                   # 훈련기관명
            }

            # 1. 지역 코드 설정
            location = user_ner.get("지역") or user_profile.get("location", "")
            if location:
                city, district = self._normalize_location(location)
                logger.info(f"[TrainingAdvisor] 정규화된 지역: city={city}, district={district}")
                
                if city:
                    area_code = self.area_codes.get(city, "")[:2]
                    logger.info(f"[TrainingAdvisor] 시/도 코드: {city} -> {area_code}")
                    params["srchTraArea1"] = area_code
                    
                    if city == "서울" and district:
                        district_code = self.area_medium_codes.get(district, "")
                        logger.info(f"[TrainingAdvisor] 구 코드: {district} -> {district_code}")
                        params["srchTraArea2"] = district_code

            # 2. 관심분야/직무 매핑
            interests = []

            # user_ner의 관심분야 처리
            ner_interests = user_ner.get("관심분야", [])
            if isinstance(ner_interests, str):
                # 쉼표로 구분된 문자열을 리스트로 변환
                interests.extend([interest.strip() for interest in ner_interests.split(",") if interest.strip()])
            elif isinstance(ner_interests, list):
                interests.extend(ner_interests)

            # user_profile의 interests 처리
            profile_interests = user_profile.get("interests", [])
            if isinstance(profile_interests, str):
                # 쉼표로 구분된 문자열을 리스트로 변환
                interests.extend([interest.strip() for interest in profile_interests.split(",") if interest.strip()])
            elif isinstance(profile_interests, list):
                interests.extend(profile_interests)

            logger.info(f"[TrainingAdvisor] user_ner: {user_ner}")
            logger.info(f"[TrainingAdvisor] 관심분야: {interests}")

            # 직무 처리
            job_type = user_ner.get("직무", "") or user_profile.get("job_type", "")
            if job_type:  # 직무가 있으면 관심분야에 추가
                interests.append(job_type)
            logger.info(f"[TrainingAdvisor] 직무 추가된 관심분야: {interests}")

            if interests:  # 관심분야가 있는 경우
                # 중복 제거 및 빈 문자열 제거
                interests = list(set(filter(None, interests)))
                target = interests[0]  # 첫 번째 관심분야 사용
                
                search_terms = [target] # 기본 검색어
                synonyms = get_job_synonyms(target)
                if synonyms:
                    search_terms.extend(synonyms)
                
                # 먼저 과정명으로만 검색 시도
                params["srchTraProcessNm"] = "|".join(search_terms)
                logger.info(f"[TrainingAdvisor] 검색어 설정: {params['srchTraProcessNm']}")
                
                # 만약 검색 결과가 없을 경우를 대비해 NCS 코드를 백업으로 저장
                ncs_backup = {}
                if target in self.interest_ncs_mapping:
                    mapping = self.interest_ncs_mapping[target]
                    ncs_backup = {
                        "srchNcs1": mapping.get("ncs1", ""),
                        "srchNcs2": mapping.get("ncs2", "")
                       
                    }
                    logger.info(f"[TrainingAdvisor] NCS 코드 설정: {ncs_backup}")
                
                # 검색 결과가 없을 경우 NCS 코드로 재시도하는 로직을 search_training_courses에 추가
                return params, ncs_backup

            logger.info(f"[TrainingAdvisor] 최종 검색 파라미터: {params}")
            return params, {}  # ncs_backup을 빈 딕셔너리로 반환

        except Exception as e:
            logger.error(f"[TrainingAdvisor] 검색 파라미터 구성 중 오류: {str(e)}")
            return params, {}  # 에러 발생 시에도 기본 파라미터와 빈 ncs_backup 반환
    
    ###############################################################################
    # 훈련과정 데이터 전처리
    ###############################################################################
    # API 응답 데이터를 표준 형식으로 변환
    # 비용 정보 정수 변환 및 포맷팅
    # 비용순 정렬 옵션 처리
    # 불필요 필드 제거
    def _format_training_courses(self, courses: List[Dict], max_count: int = 5, is_low_cost: bool = False) -> List[Dict]:
        """훈련과정 정보 포맷팅"""
        formatted_courses = []
        for course in courses:
            try:
                cost = int(course.get("courseMan", "0").replace(",", ""))
                formatted_course = {
                    "id": course.get("TRPR_ID", ""),                  # 훈련과정ID
                    "title": course.get("TRPR_NM", ""),               # 과정명
                    "institute": course.get("INST_NM", ""),           # 훈련기관명
                    "location": course.get("ADDR1", ""),              # 훈련기관 주소
                    "period": f"{course.get('TRA_START_DATE', '')}~{course.get('TRA_END_DATE', '')}", 
                    "startDate": course.get("TRA_START_DATE", ""),    # 시작일
                    "endDate": course.get("TRA_END_DATE", ""),        # 종료일
                    "cost": course.get("REAL_MAN", "0"),             # 실제 훈련비
                    "cost_value": cost,  # 정렬을 위해 추가
                    "description": course.get("TRPR_CHAP", ""),       # 과정 설명
                    "target": course.get("TRAIN_TARGET", ""),
                    "yardMan": course.get("YARD_MAN", ""),
                    "titleLink": course.get("TITLE_LINK", ""),
                    "telNo": course.get("TEL_NO", "")
                }
                formatted_courses.append(formatted_course)
            except Exception as e:
                logger.error(f"과정 포맷팅 중 오류: {str(e)}")
                continue

        # 비용순으로 정렬
        if is_low_cost:
            formatted_courses.sort(key=lambda x: x["cost_value"])
            
        # 상위 과정만 선택
        formatted_courses = formatted_courses[:max_count]
        
        # cost_value 필드 제거
        for course in formatted_courses:
            course.pop("cost_value", None)
            
        return formatted_courses

    ###############################################################################
    # 중복 제거 처리
    ###############################################################################
    # 훈련과정 목록에서 중복된 과정을 제거
    # 과정 ID 기준 중복 제거
    # 중복 제거된 과정 목록 반환
    def _deduplicate_training_courses(self, courses: List[Dict]) -> List[Dict]:
        """중복 과정 제거"""
        try:
            unique_courses = {}
            for course in courses:
                # trprId와 trprDegr를 직접 사용
                course_id = f"{course.get('id')}"
                if course_id not in unique_courses:
                    unique_courses[course_id] = course
                
            logger.info(f"[TrainingAdvisor] 중복 제거: {len(courses)} -> {len(unique_courses)}")
            return list(unique_courses.values())
        except Exception as e:
            logger.error(f"[TrainingAdvisor] 중복 제거 중 오류: {str(e)}")
            return courses

    ###############################################################################
    # 훈련과정 검색 메인 워크플로우
    ###############################################################################
    # 전체 검색 프로세스 조율(전처리 -> 검색 -> 후처리 -> 응답 생성)
    # 비용 요구사항/제외 의도 분석
    # 사용자 프로필 연동 및 이전 결과 저장
    # 최종 응답 메시지 동적 생성
    # 예외 처리 및 에러 로깅
    
    async def search_training_courses(self, query: str, user_profile: Dict = None, user_ner: Dict = None, chat_history: List[Dict] = None) -> Dict:
        """훈련과정 검색 처리"""
        try:
            logger.info(f"[TrainingAdvisor] 검색 시작 - 쿼리: {query}")
            logger.info(f"[TrainingAdvisor] 사용자 프로필: {user_profile}")
            logger.info(f"[TrainingAdvisor] NER 정보: {user_ner}")
            
            # 1. 검색 옵션 분석
            search_options = self._analyze_search_options(query, chat_history, user_ner)
            logger.info(f"[TrainingAdvisor] 검색 옵션: {search_options}")
            
            # 2. 검색 파라미터 구성
            params, ncs_backup = self._build_search_params(user_ner, user_profile)
            logger.info(f"[TrainingAdvisor] 검색 파라미터: {params}")
        
            # 3. API 호출
            courses = await self._search_courses(params)
            if not courses and ncs_backup:  # 결과가 없으면 NCS로 재시도
                params.pop("srchTraProcessNm", None)
                params.update(ncs_backup)
                courses = await self._search_courses(params)
            logger.info(f"[TrainingAdvisor] 검색된 과정 수: {len(courses) if courses else 0}")
            
            # 4. 결과 처리
            training_courses = self._process_courses(courses, search_options)
            logger.info(f"[TrainingAdvisor] 처리된 과정 수: {len(training_courses)}")
            
            if not training_courses:
                return {
                    "message": "죄송합니다. 현재 조건에 맞는 훈련과정을 찾지 못했습니다...",
                    "type": "training",
                    "trainingCourses": []
                }
            
            # 5. 응답 생성
            return self._build_response(query, training_courses, user_profile, user_ner)

        except Exception as e:
            logger.error(f"[TrainingAdvisor] 검색 중 오류: {str(e)}", exc_info=True)
            return {
                "message": f"훈련과정 검색 중 오류가 발생했습니다: {str(e)}",
                "type": "error",
                "trainingCourses": []
            }

    def _analyze_search_options(self, query: str, chat_history: List[Dict] = None, user_ner: Dict = None) -> Dict:
        """검색 옵션 분석"""
        options = {
            "is_low_cost": any(keyword in query for keyword in ["저렴한", "싼", "무료", "비용", "적은", "낮은"]),
            "is_nearby": any(keyword in query for keyword in ["가까운", "근처", "주변"]),
            "is_short_term": any(keyword in query for keyword in ["단기", "짧은", "빠른"]),
            "exclude_terms": [],
            "search_terms": []  # 빈 리스트로 초기화
        }
        
        # user_ner에서 직무와 관심분야 가져오기
        if user_ner:
            if job := user_ner.get("직무"):
                options["search_terms"].append(job)
            if interests := user_ner.get("관심분야"):
                if isinstance(interests, list):
                    options["search_terms"].extend(interests)
                elif isinstance(interests, str):
                    options["search_terms"].append(interests)
        
        # 채팅 히스토리에서 제외 의도 분석
        if chat_history:
            last_msg = chat_history[-1].get("content", "")
            if "제외" in last_msg or "빼고" in last_msg:
                options["exclude_terms"] = [word for word in last_msg.split() if word not in ["제외", "빼고", "말고", "줘", "해줘"]]
        
        return options

    async def _search_courses(self, search_params: Dict) -> List[Dict]:
        """API를 통한 과정 검색"""
        try:
            # API 응답 받기
            response = self.collector._fetch_training_list("tomorrow", search_params)
            
            # 응답 형식 검사 및 데이터 추출
            if response is None:
                logger.error("[TrainingAdvisor] API 응답이 None입니다")
                return []
            
            if isinstance(response, dict):
                # API 응답 구조: {"srchList": [...], "scn_cnt": 0, "pageSize": "50", "pageNum": "1"}
                if "srchList" in response:
                    if isinstance(response["srchList"], list):
                        # srchList가 직접 리스트인 경우
                        return response["srchList"]
                    elif isinstance(response["srchList"], dict) and "scn_list" in response["srchList"]:
                        # srchList가 dict이고 그 안에 scn_list가 있는 경우
                        return response["srchList"]["scn_list"]
                    else:
                        logger.info("[TrainingAdvisor] 검색 결과가 없습니다")
                        return []
                else:
                    logger.error(f"[TrainingAdvisor] API 응답에 srchList가 없음: {response}")
                    return []
                
            elif isinstance(response, list):
                # 리스트 형태로 직접 반환된 경우
                return response
            
            logger.error(f"[TrainingAdvisor] 처리할 수 없는 API 응답 형식: {type(response)}")
            return []
            
        except Exception as e:
            logger.error(f"[TrainingAdvisor] API 호출 중 오류: {str(e)}")
            return []

    def _process_courses(self, courses: List[Dict], search_options: Dict) -> List[Dict]:
        """검색 결과 처리 및 필터링"""
        
        try:
            # 1. 응답 데이터 포맷팅
            formatted_courses = []
            for course in courses:
                try:
                    # 비용 정보 포맷팅 (천 단위 콤마)
                    real_cost = course.get("realMan", "0")
                    total_cost = course.get("courseMan", "0")
                    
                    # 문자열에서 콤마 제거 후 숫자로 변환
                    real_cost = int(real_cost.replace(",", ""))
                    total_cost = int(total_cost.replace(",", ""))
                    
                    # 천 단위 콤마 포맷팅
                    formatted_real_cost = format(real_cost, ",")
                    formatted_total_cost = format(total_cost, ",")
                    
                    formatted_course = {
                        # API 응답의 실제 필드명으로 수정
                        "id": course.get("trprId", ""),                    # 훈련과정ID
                        "title": course.get("title", ""),                  # 과정명
                        "institute": course.get("subTitle", ""),           # 훈련기관명
                        "location": course.get("address", ""),             # 훈련기관 주소 - Pydantic 필수 필드
                        "period": f"{course.get('traStartDate', '')}~{course.get('traEndDate', '')}", 
                        "startDate": course.get("traStartDate", ""),      # 시작일
                        "endDate": course.get("traEndDate", ""),          # 종료일
                        "cost": formatted_real_cost,  # 천 단위 콤마가 적용된 실제 훈련비
                        "totalCost": formatted_total_cost,  # 천 단위 콤마가 적용된 총 훈련비               # 실제 훈련비
                        "description": course.get("contents", ""),         # 과정 설명
                        "target": course.get("trainTarget", ""),          # 훈련대상
                        "totalCost": course.get("courseMan", "0"),        # 총 훈련비
                        "yardMan": course.get("yardMan", "0"),           # 정원
                        "titleLink": course.get("titleLink", ""),         # 과정 상세 링크
                        "telNo": course.get("telNo", ""),                # 연락처
                        "address": course.get("address", ""),            # 훈련기관 주소
                        "ncsCd": course.get("ncsCd", ""),               # NCS 코드
                        "grade": course.get("grade", ""),               # 등급
                        "regCourseMan": course.get("regCourseMan", "0"), # 수강신청 인원
                        "trainingType": course.get("trainTarget", ""),    # 훈련유형
                        "trainingTarget": course.get("trainTarget", ""),  # 훈련대상
                        "employmentRate": {                               # 취업률 정보
                            "threeMonth": course.get("eiEmplRate3", "0"),
                            "sixMonth": course.get("eiEmplRate6", "0")
                        }
                    }

                    # 필수 필드 검증 - title과 location이 있는 경우만 추가
                    if formatted_course["title"] and formatted_course["location"]:
                        formatted_courses.append(formatted_course)
                    else:
                        logger.warning(f"[TrainingAdvisor] 필수 필드 누락된 과정 제외: title={formatted_course['title']}, location={formatted_course['location']}")
                        
                except ValueError as e:
                    logger.error(f"비용 포맷팅 중 오류: {str(e)}")
                    continue
                
            logger.info(f"[TrainingAdvisor] 포맷팅된 과정 수: {len(formatted_courses)}")
            
            # 2. 중복 제거
            unique_courses = self._deduplicate_training_courses(formatted_courses)
            logger.info(f"[TrainingAdvisor] 중복 제거 후 과정 수: {len(unique_courses)}")
            
            # 3. 필터링 및 정렬
            filtered_courses = self._filter_and_sort_courses(unique_courses, search_options)
            logger.info(f"[TrainingAdvisor] 최종 과정 수: {len(filtered_courses)}")
            
            return filtered_courses
            
        except Exception as e:
            logger.error(f"[TrainingAdvisor] 과정 처리 중 오류: {str(e)}")
            return []

    def _build_response(self, query: str, courses: List[Dict], user_profile: Dict, user_ner: Dict) -> Dict:
        """최종 응답 생성"""
        count = len(courses)
        location = user_ner.get("지역", "")
        interests = user_ner.get("관심분야", [])
        
        # 메시지 생성
        message_parts = []
        if location:
            message_parts.append(f"{location}지역")
        if interests:
            message_parts.append(f"'{', '.join(interests)}' 분야")
            
        message = f"{' '.join(message_parts)}의 훈련과정을 {count}개 찾았습니다." if message_parts else \
                 f"'{query}' 검색 결과 {count}개의 훈련과정이 있습니다."

        return {
            "message": message,
            "trainingCourses": courses[:5],  # 상위 5개만 반환
            "type": "training",
            "user_profile": user_profile
        }

    def _filter_and_sort_courses(self, courses: List[Dict], search_options: Dict) -> List[Dict]:
        """과정 필터링 및 정렬"""
        filtered_courses = courses[:]
        
        # 1. 직무/관심분야 기반 필터링
        if search_terms := search_options.get("search_terms", []):
            filtered_courses = [
                course for course in filtered_courses
                if any(
                    term.lower() in course.get("title", "").lower() or  # 제목에서 검색
                    term.lower() in course.get("description", "").lower() or  # 설명에서 검색
                    term.lower() in course.get("institute", "").lower() or  # 교육기관에서 검색
                    term.lower() in course.get("target", "").lower()  # 교육대상에서 검색
                    for term in search_terms
                )
            ]
            
            logger.info(f"[TrainingAdvisor] 검색어 '{search_terms}'로 필터링된 과정 수: {len(filtered_courses)}")
        
        # 2. 비용순 정렬
        if search_options.get("is_low_cost"):
            filtered_courses.sort(key=lambda x: int(x.get("cost", "0").replace(",", "")))
            logger.info("[TrainingAdvisor] 비용순으로 정렬됨")
        
        # 3. 결과가 없는 경우 원본 데이터 반환
        if not filtered_courses and courses:
            logger.warning("[TrainingAdvisor] 필터링 결과가 없어 전체 과정을 반환합니다")
            return courses[:5]  # 상위 5개만 반환
        
        return filtered_courses
