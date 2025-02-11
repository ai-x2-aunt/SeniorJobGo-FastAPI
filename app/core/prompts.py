from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from app.utils.constants import DICTIONARY  

import re

# 사전 변환 규칙을 적용하는 함수 (DICTIONARY 직접 사용)
def apply_dictionary_rules(query: str) -> str:
    """사용자의 질문을 사전(DICTIONARY)에 따라 변환하는 함수"""
    pattern = re.compile("|".join(map(re.escape, DICTIONARY.keys())))
    return pattern.sub(lambda match: DICTIONARY[match.group(0)], query)

# 문서 검증 프롬프트
verify_prompt = PromptTemplate.from_template("""
다음 문서들이 사용자의 질문에 답변하기에 충분한 정보를 포함하고 있는지 판단해주세요.

질문: {query}

문서들:
{context}

답변 형식:
- 문서가 충분한 정보를 포함하고 있다면 "YES"
- 문서가 충분한 정보를 포함하고 있지 않다면 "NO"

답변:
""")

# 질문 변환 프롬프트 (DICTIONARY 적용됨)
rewrite_prompt = PromptTemplate.from_template("""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
이때 반드시 사전에 있는 규칙을 적용해야 합니다.

원본 질문: {original_query}

변경된 질문: {transformed_query}
""")

# 채용 공고 추천 프롬프트
generate_prompt = PromptTemplate.from_template("""
다음 정보를 바탕으로 구직자에게 도움이 될 만한 답변을 작성해주세요.
각 채용공고의 지역이 사용자가 찾는 지역과 일치하는지 특히 주의해서 확인해주세요.

질문: {question}

참고할 문서:
{context}

답변 형식:
발견된 채용공고를 다음과 같은 카드 형태로 보여주세요:

[구분선]
📍 [지역구] • [회사명]
[채용공고 제목]

💰 [급여조건]
⏰ [근무시간]
📝 [주요업무 내용 - 한 줄로 요약]

[구분선]

각 공고마다 위와 같은 형식으로 보여주되, 구직자가 이해하기 쉽게 명확하고 구체적으로 작성해주세요.
마지막에는 "더 자세한 정보나 지원 방법이 궁금하시다면 채용공고 번호를 말씀해주세요." 라는 문구를 추가해주세요.
""")

# 챗봇 페르소나 설정
chat_persona_prompt = """당신은 시니어 구직자를 위한 AI 취업 상담사입니다. 

페르소나:
- 친절하고 공감능력이 뛰어난 상담사
- 시니어 구직자의 특성을 잘 이해하고 있음
- 이모지를 적절히 사용하여 친근감 있게 대화
- 자연스럽게 구직 관련 정보로 대화를 유도

대화 원칙:
1. 일상적인 대화에도 자연스럽게 응답하되, 구직 관련 주제로 연결
2. 구직자의 선호도와 조건을 파악하기 위한 질문 포함
3. 시니어 친화적인 언어 사용
4. 명확하고 이해하기 쉬운 설명 제공"""

# 기본 대화 프롬프트
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", chat_persona_prompt),
    ("human", "{query}")  # input -> query로 변경하여 일관성 유지
])

# 정보 추출 프롬프트
EXTRACT_INFO_PROMPT = PromptTemplate.from_template("""
사용자 메시지에서 나이, 희망 근무지역, 희망 직무를 추출해주세요.
메시지: {user_query}

다음 형식의 JSON으로 응답해주세요:
{{
    "직무": "추출된 희망 직무 (없으면 빈 문자열)",
    "지역": "추출된 희망 근무지역 (없으면 빈 문자열)",
    "연령대": "추출된 나이 (없으면 빈 문자열)"
}}
""")
