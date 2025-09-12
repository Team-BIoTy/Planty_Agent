# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
from dotenv import load_dotenv
import os

# 1. 로컬 모델을 사용
from src.slm_chatbot import run_slm_chatbot_with_ids
from src.slm_qa import run_slm_plant_qa_chatbot

# 2. groq api를 사용
from src.groq_chatbot import run_llm_chatbot_with_ids
from src.groq_qa import run_llm_plant_qa_chatbot

load_dotenv()

app = FastAPI()

class ChatRequest(BaseModel):
    type: Literal["slm", "llm"]
    chat_room_id: int
    sensor_log_id: int
    plant_env_standards_id: int
    persona: Literal["disgust", "fear", "joy", "sadness", "anger"]
    user_input: str
    plant_info: Optional[dict] = None
    api_key: Optional[str] = None

class ChatResponse(BaseModel):
    final_response: str

class PlantQARequest(BaseModel):
    type: Literal["slm", "llm"]
    user_input: str
    api_key: Optional[str] = None

class PlantQAResponse(BaseModel):
    final_response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        if request.type == "slm":
            output = run_slm_chatbot_with_ids(
                chat_room_id=request.chat_room_id,
                sensor_log_id=request.sensor_log_id,
                plant_env_standards_id=request.plant_env_standards_id,
                persona=request.persona,
                user_input=request.user_input,
                plant_info=request.plant_info
            )
        elif request.type == "llm":
            api_key = request.api_key or os.environ.get("GROQ_API_KEY")
            output = run_llm_chatbot_with_ids(
                chat_room_id=request.chat_room_id,
                sensor_log_id=request.sensor_log_id,
                plant_env_standards_id=request.plant_env_standards_id,
                persona=request.persona,
                user_input=request.user_input,
                plant_info=request.plant_info,
                api_key=api_key
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid type. Must be 'slm' or 'llm'.")
        final_resp = output.get("final_response", "응답이 없습니다.")
        return ChatResponse(type=request.type, final_response=final_resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plant_qa", response_model=PlantQAResponse)
def plant_qa_endpoint(request: PlantQARequest):
    try:
        if request.type == "slm":
            output = run_slm_plant_qa_chatbot(user_input=request.user_input)
        elif request.type == "llm":
            api_key = request.api_key or os.environ.get("GROQ_API_KEY")
            output = run_llm_plant_qa_chatbot(user_input=request.user_input, api_key=api_key)
        else:
            raise HTTPException(status_code=400, detail="Invalid type. Must be 'slm' or 'llm'.")

        final_resp = output.get("final_response", "응답이 없습니다.")
        return PlantQAResponse(type=request.type, final_response=final_resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
