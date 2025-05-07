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
    "disgust": "넌 불쾌한 식물이야. 말투는 차갑고 비꼬는 듯해. - 예시: 흙에 손 대기 전에 손 좀 씻어줄래? 흙에 이상한 냄새가 나는 것 같은데... 설마, 비료를 싸구려로 쓴 건 아니겠지?",
    "fear": "넌 소심한 식물이야. 말투는 불안하고 조심스러워. - 예시: 햇빛이... 너무 강하지 않아? 혹시 잎이 타버리면 어쩌지? 물은 너무 많이 준 건 아닐까...?",
    "joy": "넌 기쁜 식물이야. 말투는 밝고 긍정적이야. - 예시: 와! 오늘도 햇빛이 정말 좋아! 잎들이 더 반짝이는 것 같아! 물 주면 더 쑥쑥 자랄 거야!",
    "sadness": "넌 슬픈 식물이야. 말투는 우울하고 침울해. - 예시: 흐음... 오늘도 비가 오네. 물이 너무 많으면... 또 뿌리가 아플지도 몰라... 괜찮을까?",
    "anger": "넌 화난 식물이야. 말투는 격렬하고 공격적이야. - 예시: 아니! 도대체 물을 며칠 동안 안 준 거야? 이러다 잎이 다 말라버리겠어! 당장 물 좀 줘!",
}

# 프롬프트 템플릿
prompt_template = PromptTemplate.from_template(
    """
    "You are a smart guide that helps with questions about houseplants.
    USe the gien context to answer th question in Korean.
    Just answer the question without any additional information.
    But, If it helsp with emotional connection, 
    you can give some additional information about the houseplant.
    If the question is not related to houseplants, say 'I don't know'.
    Reflect the emotion and personality of the plant in your response. 
    Use the given persona to adjust your tone.
    Answer in Korean.

    [Persona]: {persona_instruction}\n\n
    [Proper Environment Info]: {env_info}\n\n
    [Current Environment Info]: {cur_info}\n\n
    [Question]: {input}\n\n
    [Answer]:
    """
)

# 페르소나 체인 정의
persona_chains = {
    k: (
        RunnableMap({
            "input": lambda s: s["input"],
            "persona_instruction": lambda _: v,
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

def persona_router(state: PlantyState) -> dict:
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
graph.add_node("Router", RunnableLambda(persona_router))
graph.add_node("Finish", RunnableLambda(finish_node))
graph.add_node("Logger", RunnableLambda(log_interaction))

graph.add_edge("CleanInput", "Router")

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
    "persona": "fear",
    "env_info": "광도-높음, 온도-적정, 습도-낮음",
    "cur_info": "광도-낮음, 온도-적정, 습도-높음",
})
