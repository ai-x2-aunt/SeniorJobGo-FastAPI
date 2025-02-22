"""
채팅 관련 API 라우터입니다.
"""

from fastapi import APIRouter, HTTPException
from bson import ObjectId
from .models import ChatModel, UserModel
from .database import db

router = APIRouter()

# 채팅 조회 (페이징 처리 추가 예정)
@router.get("/get/limit/{_id}")
async def get_chats_by_user(_id: str, end: int, limit: int):
    # 우선 대화 기록을 전부 불러오게 하고 페이징 처리를 추후에 구현할 예정
    user = await db.users.find_one({"_id": ObjectId(_id)})

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if end == -1:
        end = len(user["messages"])

    start = max(end - limit, 0)

    chatList = user["messages"] if user else []

    if len(chatList) == 0:
        return {"index": end, "messages": []}
    if start > len(chatList):
        return {"index": start, "messages": []}
    return {"index": max(start, 0), "messages": chatList[start:end]}

    # 기존 코드 (채팅 기록을 따로 빼두었을 경우)
    # - 처음에 채팅 기록을 따로 빼두었을 경우 사용하는 코드
    # - 현재는 채팅 기록을 따로 빼두지 않고 모든 채팅 기록을 한 번에 불러오기 때문에 사용하지 않음
    # - 코드에 대한 설명은 notion에 남겨두었습니다. (기타 참고자료 > MongoDB 페이징 처리)
    #   - https://www.notion.so/Backend-2-MongoDB-1f64eb547b2342fc84eb373391c92c31
    # skip = (page - 1) * page_size  # 건너뛸 문서 수

    # chats = await db.chats.find({"id": user_id, "provider": provider}) \
    #     .skip(skip) \
    #     .limit(page_size) \
    #     .to_list(page_size)

# 모든 채팅 조회
# 사용하지 않는 메서드이나 참고를 위해 남겨둠.
@router.get("/get/all/{_id}")
async def get_all_chats(_id: str):
    user = await db.users.find_one({"_id": ObjectId(_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user["messages"]

# 모든 채팅 삭제
# 데이터베이스에서 대화를 삭제하고 싶다면 사용하는 메서드
# 사용하지 않는 메서드이나 참고를 위해 남겨둠.
@router.delete("/delete/all/{_id}")
async def delete_all_chats(_id: str) -> bool:
    user = await db.users.find_one({"_id": ObjectId(_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        await db.users.update_one({"_id": ObjectId(_id)}, {"$set": {"messages": []}})
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add/{_id}")
async def add_chat_message(user: UserModel, user_message: str, bot_message: str):
    chat_index = len(user.get("messages", []))

    user_message_model: ChatModel = {
        "role": "user",
        "content": user_message,
        "index": chat_index
    }
    
    bot_message_model: ChatModel = {
        "role": "bot",
        "content": bot_message,
        "index": chat_index + 1
    }

    await db.users.update_one(
        {"_id": ObjectId(user.get("_id"))}, 
        {"$push": {"messages": {"$each": [user_message_model, bot_message_model]}}}
    )