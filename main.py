from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
import uvicorn

from your_module import app as chatbot_app, fetch_row_by_id  # 기존 코드 모듈화 필요

app = FastAPI()

class ChatRequest(BaseModel):
    room_id: int
    persona: Literal["disgust", "fear", "joy", "sadness", "anger"]

@app.post("/chat")
def chat(req: ChatRequest):
    # DB에서 채팅 메시지 및 상태 정보 가져오기
    chat_message = fetch_row_by_id("chat_messages", req.room_id) or {}
    iot_device = fetch_row_by_id("iot_device", req.room_id) or {}
    plant_env_standard = fetch_row_by_id("plant_env_standards", req.room_id) or {}

    if not chat_message:
        raise HTTPException(status_code=404, detail="Chat message not found.")

    env_info_str = (
        f"최대 습도: {plant_env_standard.get('max_humidity', '정보 없음')}, "
        f"최대 광도: {plant_env_standard.get('max_light', '정보 없음')}, "
        f"최대 온도: {plant_env_standard.get('max_temperature', '정보 없음')}, "
        f"최소 습도: {plant_env_standard.get('min_humidity', '정보 없음')}, "
        f"최소 광도: {plant_env_standard.get('min_light', '정보 없음')}, "
        f"최소 온도: {plant_env_standard.get('min_temperature', '정보 없음')}"
    )

    cur_info_str = (
        f"IoT 기기 상태 - 모델명: {iot_device.get('model_name', '정보 없음')}, "
        f"상태: {iot_device.get('status', '정보 없음')}"
    )

    result = chatbot_app.invoke({
        "input": chat_message.get('message', '안녕! 오늘 기분이 어때?'),
        "persona": req.persona,
        "env_info": env_info_str,
        "cur_info": cur_info_str,
    })

    return {"response": result.get("final_response", "응답이 없습니다.")}

# vllm
# import requests

# def generate_with_vllm(prompt: str) -> str:
#     response = requests.post("http://localhost:8000/generate", json={
#         "prompt": prompt,
#         "max_tokens": 512,
#         "temperature": 0.7
#     })
#     return response.json()["text"]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

