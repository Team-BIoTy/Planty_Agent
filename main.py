# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional

# 1. 로컬 모델을 사용
# from src.slm_chatbot import run_chatbot_with_ids
# from src.slm_qa import run_plant_qa_chatbot

# 2. groq api를 사용
from src.groq_chatbot import run_chatbot_with_ids
from src.groq_qa import run_plant_qa_chatbot

app = FastAPI()

class ChatRequest(BaseModel):
    chat_room_id: int
    sensor_log_id: int
    plant_env_standards_id: int
    persona: Literal["disgust", "fear", "joy", "sadness", "anger"]
    user_input: str
    plant_info: Optional[dict] = None

class ChatResponse(BaseModel):
    final_response: str

class PlantQARequest(BaseModel):
    user_input: str

class PlantQAResponse(BaseModel):
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
            plant_info=request.plant_info
        )
        final_resp = output.get("final_response", "응답이 없습니다.")
        return ChatResponse(final_response=final_resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plant_qa", response_model=PlantQAResponse)
def plant_qa_endpoint(request: PlantQARequest):
    try:
        output = run_plant_qa_chatbot(user_input=request.user_input)
        final_resp = output.get("final_response", "응답이 없습니다.")
        return PlantQAResponse(final_response=final_resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
