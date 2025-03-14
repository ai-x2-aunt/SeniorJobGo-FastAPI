from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    user_message: str
    user_profile: Optional[dict] = None
    session_id: Optional[str] = None

class JobPosting(BaseModel):
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
    applicationPeriod: str
    supplytype: str
    contact: str
    url: str

class MealPosting(BaseModel):
    name: str
    address: str
    phone: str
    operatingHours: str
    targetGroup: str
    description: str
    latitude: float = 0.0  # 위도 필드 추가
    longitude: float = 0.0  # 경도 필드 추가

class ChatResponse(BaseModel):
    type: str  # 'list' 또는 'detail'
    message: str
    jobPostings: List[JobPosting]
    trainingCourses: List[TrainingCourse] = []  # 훈련과정 정보 추가
    policyPostings: List[PolicyPosting] = []  # 정책 정보 추가
    mealPostings: List[MealPosting] = []  # 식사 정보 추가
    user_profile: Optional[dict] = None
    processingTime: float = 0  # 처리 시간 추가

