from pydantic import BaseModel
from typing import Dict, List, Optional

class ChatRequest(BaseModel):
    """채팅 요청 스키마"""
    user_message: str
    user_profile: Optional[dict] = None
    session_id: Optional[str] = None

class JobPosting(BaseModel):
    """채용 공고 스키마"""
    id: int
    title: str
    company: str
    location: str
    salary: str
    workingHours: str
    description: str
    phoneNumber: str
    deadline: str
    requiredDocs: str
    hiringProcess: str
    insurance: str
    jobCategory: str
    jobKeywords: str
    posting_url: str

class TrainingCourse(BaseModel):
    """훈련과정 스키마"""
    id: str
    title: str
    institute: str
    location: str
    period: str
    startDate: str
    endDate: str
    cost: str
    description: str
    target: Optional[str] = None
    yardMan: Optional[str] = None
    titleLink: Optional[str] = None
    telNo: Optional[str] = None


class PolicyPosting(BaseModel):
    source: str
    title: str
    target: str
    content: str
    applyMethod: str
    contact: str
    url: str

class MealPosting(BaseModel):
    name: str
    address: str
    phone: str
    operatingHours: str
    targetGroup: str
    description: str 


class ChatResponse(BaseModel):
    """채팅 응답 스키마"""
    type: str  # 'list' 또는 'detail'
    message: str
    jobPostings: List[JobPosting]
    trainingCourses: List[TrainingCourse] = []  # 훈련과정 정보 추가
    policyPostings: List[PolicyPosting] = []  # 정책 정보 추가
    mealPostings: List[MealPosting] = []  # 식사 정보 추가
    user_profile: Optional[dict] = None
    processingTime: float = 0  # 처리 시간 추가


