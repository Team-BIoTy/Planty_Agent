# chatbot_app.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# # groq_chatbot.py
# from base.chatbot import PersonaChatbot

# main.py
from src.base.chatbot import PersonaChatbot

############################ 환경 설정 ############################
load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

############################ 실행 함수 ############################
def run_llm_chatbot_with_ids(
    chat_room_id: int,
    sensor_log_id: int,
    plant_env_standards_id: int,
    persona: str = "joy",
    user_input: str = "",
    plant_info: dict | None = None
) -> dict:
    
    # 모델 로드
    lm = ChatGroq(
        model="gemma2-9b-it", # Groq 모델 이름
        temperature=0.7,
        max_tokens=256,
    )

    bot = PersonaChatbot(lm, type="LLM")

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

def run_llm_chatbot_with_direct_data(
    nickname: str,
    env_info_dict: dict,
    cur_info_dict: dict,
    chat_log: str,
    persona: str = "joy",
    user_input: str = ""
) -> dict:

    # 모델 로드
    lm = ChatGroq(
        model="gemma2-9b-it", # Groq 모델 이름
        temperature=0.7,
        max_tokens=256,
    )

    bot = PersonaChatbot(lm, type="LLM")
    
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
    
#     num_runs = 1
#     times = []

#     for i in range(num_runs):
#         start_time = time.time()

#         result = run_llm_chatbot_with_direct_data(
#             nickname="플로라",
#             env_info_dict=env_info,
#             cur_info_dict=cur_info,
#             chat_log=chat_log,
#             persona="joy",
#             user_input="몬스테라에 대해서 설명해줘"
#         )

#         elapsed = time.time() - start_time
#         times.append(elapsed)

#         print(f"[{i+1}/{num_runs}] 소요 시간: {elapsed:.2f}초")
#         print("응답:", result.get("final_response", "응답이 없습니다."))
#         print("="*40)

#     avg_time = sum(times) / num_runs
#     print(f"\n=== 평균 소요 시간: {avg_time:.2f}초 ===")
