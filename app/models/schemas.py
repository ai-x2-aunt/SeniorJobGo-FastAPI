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
    
class ChatResponse(BaseModel):
    type: str  # 'list' 또는 'detail'
    message: str
    jobPostings: List[JobPosting]
    trainingCourses: List[TrainingCourse] = []  # 훈련과정 정보 추가
    user_profile: Optional[dict] = None
    processingTime: float = 0  # 처리 시간 추가 