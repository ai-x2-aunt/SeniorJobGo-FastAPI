"""
회원 인증 관련 라우트 정의
"""

from fastapi import APIRouter, HTTPException, Request, Response
from datetime import datetime
import bcrypt
import uuid
import httpx
import os
from .database import db
from .models import UserModel

router = APIRouter()

# 비밀번호 해싱 함수
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# 비밀번호 검증 함수
def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

def set_cookie(response: Response, id: str, provider: str):
    max_age = 60*60*24*30
    response.set_cookie(key="sjgid", value=id, max_age=max_age)
    response.set_cookie(key="sjgpr", value=provider, max_age=max_age)

@router.get("/check")
async def check_cookie(request: Request) -> bool:
    _id = request.cookies.get("sjgid")
    provider = request.cookies.get("sjgpr")
    user = await db.users.find_one({"_id": _id, "provider": provider})
    return user is not None

# 사용자 회원가입 (Signup)
@router.post("/signup")
async def signup_user(request: Request, response: Response):
    try:
        data = await request.json()
        
        # userId를 id로 변환
        user_id = data.get("userId")  # "userId"로 변경
        user = await db.users.find_one({"id": user_id})

        if user:
            raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")

        # UserModel 생성 시 id 필드명 사용
        user = UserModel(
            id=user_id,  
            password=hash_password(data.get("password")),
            provider="local"
        )

        user_dict = user.model_dump()
        result = await db.users.insert_one(user_dict)
        set_cookie(response, str(result.inserted_id), "local")
        return {**user_dict, "_id": str(result.inserted_id)}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="회원가입에 실패했습니다.")

# 사용자 로그인 (Login)
@router.post("/login")
async def login_user(request: Request, response: Response) -> bool:
    data = await request.json()
    user_id = data.get("user_id")
    password = data.get("password")
    provider = data.get("provider")

    if provider == "local":
        user = await db.users.find_one({"id": user_id, "provider": "local"})
        if user:
            if verify_password(password, user["password"]):
                _id = str(user["_id"])
                await db.users.update_one({"_id": _id}, {"$set": {"last_login": datetime.now()}})
                set_cookie(response, str(_id), "local")
                return True

    raise HTTPException(status_code=401, detail="Invalid credentials")

# 사용자 아이디 중복 확인 (Check ID)
@router.post("/check-id")
async def check_id(request: Request):
    data = await request.json()
    user_id = data.get("id")
    user = await db.users.find_one({"id": user_id})
    return {"is_duplicate": user is not None}

# 비회원 로그인 (Guest Login)
@router.post("/login/guest")
async def guest_login(response: Response):
    user = UserModel(id=str(uuid.uuid4()), password=hash_password("guest"), provider="none")

    user_dict = user.model_dump()
    result = await db.users.insert_one(user_dict)
    set_cookie(response, str(result.inserted_id), "none")
    return {**user_dict, "_id": str(result.inserted_id)}

# 비회원 전부 삭제
@router.delete("/delete/guest")
async def delete_guest():
    await db.users.delete_many({"provider": "none"})
    return {"message": "All guest user deleted"}

# 사용자 카카오 로그인 (Kakao Login)
@router.get("/kakao/callback")
async def kakao_callback(code: str, response: Response):
    """ 카카오 OAuth 인증 후 액세스 토큰 요청 """
    token_url = "https://kauth.kakao.com/oauth/token"

    KAKAO_CLIENT_ID = os.getenv("KAKAO_CLIENT_ID")
    KAKAO_CLIENT_SECRET = os.getenv("KAKAO_CLIENT_SECRET")
    KAKAO_REDIRECT_URI = os.getenv("KAKAO_REDIRECT_URI")

    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "client_id": KAKAO_CLIENT_ID,
                "client_secret": KAKAO_CLIENT_SECRET,
                "redirect_uri": KAKAO_REDIRECT_URI,
                "code": code,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        token_data = token_response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to get access token")

        # 사용자 정보 가져오기
        user_info_url = "https://kapi.kakao.com/v2/user/me"
        user_info_response = await client.get(
            user_info_url, headers={"Authorization": f"Bearer {access_token}"}
        )

        user_info = user_info_response.json()
        user_id = str(user_info["id"])
        _id = None

        user = await db.users.find_one({"id": user_id})
        # 만약 카카오 로그인 시 이미 회원가입이 되어있는 사용자라면 로그인 처리
        if user:
            _id = str(user["_id"])
        else:
            # kakao_login_info.json 파일에 있는 정보를 사용하여 UserModel 생성
            # 주석 처리된 부분은 UserModel에 없는 필드이므로 추후 추가한다면 해당되는 부분을 주석 해제하여 사용할 수 있음
            user_info = UserModel(
                id=user_id,
                password=hash_password("kakao"),
                provider="kakao",
                name=user_info["kakao_account"]["name"],
                # email=user_info["kakao_account"]["email"] if user_info["kakao_account"]["email_needs_agreement"] else None,
                # phone=user_info["kakao_account"]["phone_number"],
                gender=user_info["kakao_account"]["gender"],
                # age=user_info["kakao_account"]["age_range"],
                birth_year=int(user_info["kakao_account"]["birthyear"])
            )

            user_dict = user_info.model_dump()
            user = await db.users.insert_one(user_dict)
            _id = str(user.inserted_id)

        # 사용자 정보 출력 테스트 코드
        response = Response()
        set_cookie(response, _id, "kakao")
        response.headers["Location"] = "http://localhost:5173/chat"
        response.status_code = 303 # 오류가 아니라 리다이렉트 코드에요

        return response
