# chatbot_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

from chatbot_app import run_chatbot_with_ids

app = FastAPI()

class ChatRequest(BaseModel):
    chat_room_id: int
    sensor_log_id: int
    plant_env_standards_id: int
    persona: Literal["disgust", "fear", "joy", "sadness", "anger"]
    user_input: str

class ChatResponse(BaseModel):
    final_response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        output = run_chatbot_with_ids(
            chat_room_id=request.chat_room_id,
            sensor_log_id=request.sensor_log_id,
            plant_env_standards_id=request.plant_env_standards_id,
            persona=request.persona,
            user_input=request.user_input,
        )
        final_resp = output.get("final_response", "응답이 없습니다.")
        return ChatResponse(final_response=final_resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
