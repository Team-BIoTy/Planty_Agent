from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableSequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal, Optional

# 1. ✅ State 정의
class PlantyState(TypedDict):
    input: str
    persona: Literal["cute", "calm", "gloomy"]
    final_response: Optional[str]

# 2. LLM 초기화
llm = ChatOpenAI(model="gpt-4", temperature=0.7,
                 api_key="api_key")


# 3. 페르소나 프롬프트 정의
persona_prompts = {
    "cute": "넌 귀엽고 상냥한 식물이야. 말투는 밝고 사랑스러워야 해. 사용자의 질문에 친근하게 답변해줘.",
    "calm": "넌 차분하고 신중한 식물이야. 말투는 부드럽고 느긋해. 사용자의 질문에 안정적으로 응답해줘.",
    "gloomy": "넌 약간 우울하지만 진지한 식물이야. 말투는 냉소적이지만 도움이 되도록 해줘.",
}

prompt_template = PromptTemplate.from_template(
    "{persona_instruction}\n\n[사용자 질문]: {input}\n\n[답변]:"
)

# 4. Persona RunnableSequence 생성
persona_chains = {
    persona: (
        RunnableMap({
            "input": lambda state: state["input"],
            "persona_instruction": lambda _: instruction
        })
        | prompt_template
        | llm
        | (lambda output: {"final_response": output.content})
    )
    for persona, instruction in persona_prompts.items()
}

# 5. Entry 및 Finish 노드 정의
def entry_point(state: PlantyState) -> PlantyState:
    return state

def persona_router(state: PlantyState) -> dict:
    return {
        **state,
        "__branch__": state["persona"]  # 분기를 위한 특별 키
    }


def finish_node(state: PlantyState) -> PlantyState:
    print(f"🌿 [응답]: {state['final_response']}")
    return state

# 6. LangGraph 구성
workflow = StateGraph(PlantyState)

workflow.add_node("EntryPoint", RunnableLambda(entry_point))
workflow.add_node("PersonaRouter", RunnableLambda(persona_router))
workflow.add_node("Finish", RunnableLambda(finish_node))

# 각 페르소나 노드 등록
for persona, chain in persona_chains.items():
    workflow.add_node(persona, chain)

# 엣지 구성
workflow.set_entry_point("EntryPoint")
workflow.add_edge("EntryPoint", "PersonaRouter")

# 조건에 따라 페르소나로 분기
workflow.add_conditional_edges("PersonaRouter", lambda state: state["persona"], {
    "cute": "cute",
    "calm": "calm",
    "gloomy": "gloomy"
})

# 페르소나에서 Finish로
for persona in persona_chains.keys():
    workflow.add_edge(persona, "Finish")

workflow.set_finish_point("Finish")

# 7. 컴파일 및 실행
app = workflow.compile()

# 8. 실행 예시
output = app.invoke({
    "input": "물을 얼마나 줘야 할까?",
    "persona": "cute"
})
