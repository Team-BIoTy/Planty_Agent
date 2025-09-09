# chatbot_app.py
import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline

# slm_chatbot.py
# from base.chatbot import PersonaChatbot

# main.py
from src.base.chatbot import PersonaChatbot

############################ 환경 설정 ############################

load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

############################ 모델 로딩 ############################

model_path = "./HyperCLOVAX-Local"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

hf_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    device=0,
    max_new_tokens=256,
    return_full_text=False,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
)

lm = HuggingFacePipeline(pipeline=hf_pipeline, model_kwargs={"temperature": 0.7,
                                                             "stopping_criteria": [[tokenizer.eos_token_id]]})

############################ 실행 함수 ############################
# 데이터베이스에서 정보를 가져와 챗봇을 실행하는 함수
def run_slm_chatbot_with_ids(
    chat_room_id: int,
    sensor_log_id: int,
    plant_env_standards_id: int,
    persona: str = "joy",
    user_input: str = "",
    plant_info: dict | None = None
) -> dict:
    bot = PersonaChatbot(lm)

    params = {
        "chat_room_id": chat_room_id,
        "sensor_log_id": sensor_log_id,
        "plant_env_standards_id": plant_env_standards_id,
        "persona": persona,
        "user_input": user_input,
        "plant_info": plant_info
    }

    response = bot.run(**params)
    return response

def run_slm_chatbot_with_direct_data(
    nickname: str,
    env_info_dict: dict,
    cur_info_dict: dict,
    chat_log: str,
    persona: str = "joy",
    user_input: str = ""
) -> dict:
    bot = PersonaChatbot(lm)
    
    params = {
        "nickname": nickname,
        "env_info_dict": env_info_dict,
        "cur_info_dict": cur_info_dict,
        "chat_log": chat_log,
        "persona": persona,
        "user_input": user_input
    }

    response = bot.run_direct_data(**params)

    return response

# ############################ 실행 예시 ############################

# if __name__ == "__main__":
#     # 데이터베이스에 저장된 정보를 기반으로 챗봇을 실행하는 예시
#     # result = run_chatbot_with_ids(chat_room_id=1, sensor_log_id=1, plant_env_standards_id=1, persona="joy", user_input="안녕, 오늘 날씨 어때?")
#     # print("=== 챗봇 응답 ===")
#     # print(result.get("final_response", "응답이 없습니다."))

#     # 테스트용 데이터
#     env_info = {
#         "max_humidity": 80,
#         "max_light": 15000,
#         "max_temperature": 30,
#         "min_humidity": 40,
#         "min_light": 5000,
#         "min_temperature": 15
#     }

#     cur_info = {
#         "temperature": 28,
#         "humidity": 55,
#         "light": 12000,
#         "timestamp": "2025-05-29 14:00:00"
#     }

#     chat_log = """안녕하세요! 오늘은 물을 잘 줬어요.
#         기분이 어때요?

#         오늘 햇빛이 많이 들어왔어요.
#         그래서인지 잎이 더 반짝거려요.

#         아침에는 조금 추웠는데, 지금은 따뜻해졌네요.
#         혹시 오늘도 음악 틀어줄 수 있나요?"""
    
#     import time
    
#     # 속도 테스트
#     start_time = time.time()

#     num_runs = 10
#     times = []

#     for i in range(num_runs):
#         start_time = time.time()

#         result = run_slm_chatbot_with_direct_data(
#             nickname="플로라",
#             env_info_dict=env_info,
#             cur_info_dict=cur_info,
#             chat_log=chat_log,
#             persona="joy",
#             user_input="오늘 기분이 어때?"
#         )

#         elapsed = time.time() - start_time
#         times.append(elapsed)

#         print(f"[{i+1}/{num_runs}] 소요 시간: {elapsed:.2f}초")
#         print("응답:", result.get("final_response", "응답이 없습니다."))
#         print("="*40)

#     avg_time = sum(times) / num_runs
#     print(f"\n=== 평균 소요 시간: {avg_time:.2f}초 ===")

