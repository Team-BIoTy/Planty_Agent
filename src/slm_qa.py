import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from src.base.qa import PlantQAChatbot

# ======================== 환경 설정 ========================
load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

# ======================== 답변 함수 ========================
def run_plant_qa_chatbot(user_input: str):
    # 모델 로딩
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

    lm = HuggingFacePipeline(pipeline=hf_pipeline, model_kwargs={"temperature": 0.7})

    bot = PlantQAChatbot(lm)
    result = bot.run(user_input)
    return result

# ======================== 실행 예시 ========================
if __name__ == "__main__":
    import time

    # num_runs = 10
    # times = []

    questions = [
        "겨울에 기르기 좋은 식물 3가지만 추천해줘",
        # "실내식물을 기르기 위해 필요한 조건은 뭐야?",
        # "가울테리아에 대해서 설명해줘",
        # "물을 자주 주지 않아도 잘 자라는 식물은 어떤 게 있을까?",
        # "만약 식물에 물을 너무 많이 줬으면 어떻게 해야 할까?",
        # "산세베리아를 키울 때 주의할 점을 알려줘",
        # "무궁화에 대해 설명해봐",
        # "분갈이를 위해서 사야하는 화분의 크기는 몇 인치가 적당할까?",
        # "산세베리아 분갈이를 위해서 사야하는 화분의 크기는 몇 인치가 적당할까?"
    ]

    for question in questions:    
        print("=" * 40)
        start_time = time.time()
        
        result = run_plant_qa_chatbot(question)
        
        elapsed = time.time() - start_time
        # times.append(elapsed)

        print("=" * 40)
        print(f"질문: {question}")
        print(f"소요 시간: {elapsed:.2f}초")
        print("응답:", result.get("final_response", "응답이 없습니다."))
        print("="*40)

    # avg_time = sum(times) / num_runs
    # print(f"\n=== 평균 소요 시간: {avg_time:.2f}초 ===")