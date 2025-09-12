import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# groq_qa.py
# from base.qa import PlantQAChatbot

# main.py
from src.base.qa import PlantQAChatbot

# ======================== 환경 설정 ========================
load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

# ======================== 답변 함수 ========================
def run_llm_plant_qa_chatbot(user_input: str, api_key: str | None = None):
    # 모델 로딩
    lm = ChatGroq(
        model="gemma2-9b-it",
        temperature=0.7,
        max_tokens=512,
        api_key=api_key
    )

    bot = PlantQAChatbot(lm)
    result = bot.run(user_input)
    return result

# ======================== 실행 예시 ========================
# if __name__ == "__main__":
#     import time

#     # num_runs = 10
#     # times = []

#     questions = [
#         "여름에 기르기 좋은 식물 3가지만 추천해줘",
#         # "실내식물을 기르기 위해 필요한 조건은 뭐야?",
#         # "몬스테라에 대해서 설명해줘",
#         # "물을 자주 주지 않아도 잘 자라는 식물은 어떤 게 있을까?",
#         # "만약 식물에 물을 너무 많이 줬으면 어떻게 해야 할까?",
#         # "산세베리아를 키울 때 주의할 점을 알려줘",
#         # "무궁화에 대해 설명해봐",
#         # "분갈이를 위해서 사야하는 화분의 크기는 몇 인치가 적당할까?",
#         # "산세베리아 분갈이를 위해서 사야하는 화분의 크기는 몇 인치가 적당할까?"
#     ]

#     for question in questions:    
#         print("=" * 40)
#         start_time = time.time()
        
#         result = run_llm_plant_qa_chatbot(question)
        
#         elapsed = time.time() - start_time
#         # times.append(elapsed)

#         print("=" * 40)
#         print(f"질문: {question}")
#         print(f"소요 시간: {elapsed:.2f}초")
#         print("응답:", result.get("final_response", "응답이 없습니다."))
#         print("="*40)

#     # avg_time = sum(times) / num_runs
#     # print(f"\n=== 평균 소요 시간: {avg_time:.2f}초 ===")