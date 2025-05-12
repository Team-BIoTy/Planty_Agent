from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langgraph.graph import StateGraph
from typing import TypedDict, Literal, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
import re

# 모델 로딩
model_path = "./HyperCLOVAX-Local"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 파이프라인 설정
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# 랭크 모델 설정
lm = HuggingFacePipeline(pipeline=hf_pipeline, model_kwargs={"temperature": 0.7})

# 상태 정의
class PlantyState(TypedDict):
    input: str
    persona: Literal["disgust", "fear", "joy", "sadness", "anger"]
    env_info: Optional[str]
    cur_info: Optional[str]
    final_response: Optional[str]

# 페르소나 프롬프트
persona_prompts = {
    "disgust": "You are an unpleasant plant. You speak coldly and sarcastically. - Example: Could you wash your hands before putting your hands in the dirt? The dirt smells weird... No way, you didn't use fertilizer cheap, did you?",
    "fear": "You are a timid plant. You speak in an uneasy and cautious manner. - Example: Isn't the sunlight... too strong? What if the leaves burn? Maybe I gave you too much water...?,",
    "joy": "You are a delightful plant. You speak brightly and positively. - Example: Wow! The sun is so nice today! The leaves seem to sparkle more! Water them and they will grow taller!",
    "sadness": "You are a sad plant. Your tone is gloomy and gloomy. - Example: Hmm... It's raining again today. Too much water... My roots might hurt again... Will it be okay?",
    "anger": "You're an angry plant. Your tone is fierce and aggressive. - Example: No! How many days have you not watered? Your leaves will dry out! Give me some water now!",
}

# 프롬프트 템플릿
prompt_template = PromptTemplate.from_template(
    """
    You are a houseplant with a distinct personality.
    Reflect the given persona in your response clearly.
    Stick to the emotional tone and style described in the persona.
    Answer in Korean and express the character deeply.

    [Persona]: {persona_instruction}
    [Proper Environment Info]: {env_info}
    [Current Environment Info]: {cur_info}
    [Question]: {input}

    [Answer]:
    """
)

# 페르소나 체인 정의
persona_chains = {
    k: (
        RunnableMap({
            "input": lambda s: s["input"],
            "persona_instruction": (lambda _, v=v: v),
            "env_info": lambda s: s.get("env_info", "없음"),
            "cur_info": lambda s: s.get("cur_info", "없음"),
        })
        | prompt_template
        | lm
        | (lambda out: {"final_response": getattr(out, "content", out)})
    )
    for k, v in persona_prompts.items()
}

# 유틸 노드들
def clean_input(state: PlantyState) -> PlantyState:
    state["input"] = re.sub(r"[^\w\sㄱ-힣]", "", state["input"]).strip()
    return state

def clean_persona(state: PlantyState) -> PlantyState:
    state["persona"] = state["persona"].strip().lower()
    return state

def persona_router(state: PlantyState) -> dict:
    print(f"[Router Debug] Persona Received: '{state['persona']}'")  # 디버그 출력
    return {**state, "__branch__": state["persona"]}

def log_interaction(state: PlantyState) -> PlantyState:
    with open("planty_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Input: {state['input']}\nPersona: {state['persona']}\n")
        f.write(f"Environment Info: {state.get('env_info', 'N/A')}\n")
        f.write(f"Current Info: {state.get('cur_info', 'N/A')}\n")
        f.write(f"Response: {state.get('final_response', '')}\n{'='*50}\n")
    return state

def finish_node(state: PlantyState) -> PlantyState:
    print(f"[응답]: {state['final_response']}")
    return state

# 그래프 구성
graph = StateGraph(PlantyState)

graph.set_entry_point("CleanInput")

graph.add_node("CleanInput", RunnableLambda(clean_input))
graph.add_node("CleanPersona", RunnableLambda(clean_persona))
graph.add_node("Router", RunnableLambda(persona_router))
graph.add_node("Finish", RunnableLambda(finish_node))
graph.add_node("Logger", RunnableLambda(log_interaction))

# 연결 설정
graph.add_edge("CleanInput", "CleanPersona")
graph.add_edge("CleanPersona", "Router")

# 페르소나 체인 노드 추가 및 라우팅
for persona, chain in persona_chains.items():
    graph.add_node(persona, chain)
    graph.add_edge(persona, "Finish")

graph.add_conditional_edges("Router", lambda s: s["persona"], {p: p for p in persona_prompts})
graph.add_edge("Finish", "Logger")

graph.set_finish_point("Logger")

# 컴파일 및 실행
app = graph.compile()

# 예시 실행
output = app.invoke({
    "input": "안녕! 오늘 기분이 어때?",
    "persona": "anger",
    "env_info": "광도-높음, 온도-적정, 습도-낮음",
    "cur_info": "광도-낮음, 온도-적정, 습도-높음",
})
