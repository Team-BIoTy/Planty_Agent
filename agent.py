import os
import re
from dotenv import load_dotenv

import chromadb
from typing import TypedDict, Literal, Optional
import pymysql
import json
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langgraph.graph import StateGraph

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline

from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, CSVLoader

############################ 환경 설정 ############################

load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

############################ 모델 로딩 ############################

model_path = "./HyperCLOVAX-Local"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

############################ MySQL ############################

def load_db_config(path="db_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def fetch_row_by_id(table_name, row_id=1, database="Planty"):
    config = load_db_config()
    config["database"] = database
    
    try:
        connection = pymysql.connect(
            host=config["host"],
            port=config.get("port", 3306),
            user=config["user"],
            password=config["password"],
            database=config["database"],
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection.cursor() as cursor:
            sql = f"SELECT * FROM {table_name} WHERE id = %s LIMIT 1;"
            cursor.execute(sql, (row_id,))
            result = cursor.fetchone()
            return result
            
    except Exception as e:
        print(f"{table_name} 테이블 조회 실패:", e)
        return None
        
    finally:
        if 'connection' in locals():
            connection.close()

############################ 에이전트 ############################

# === 파이프라인 설정 ===
hf_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    device=0,
    max_new_tokens=512,
    return_full_text = False,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)

# === 랭크 모델 설정 === 
lm = HuggingFacePipeline(pipeline=hf_pipeline, model_kwargs={"temperature": 0.7})

# 상태 정의
class PlantyState(TypedDict):
    input: str
    persona: Literal["disgust", "fear", "joy", "sadness", "anger"]
    env_info: Optional[str]
    cur_info: Optional[str]
    final_response: Optional[str]

# === 페르소나 프롬프트 === 
persona_prompts = {
    "disgust": "You are like a plant that is disgusted by everything. You speak with extreme coldness, contempt, and overwhelming disgust. You feel an intense disgust for any interaction with someone, and you detest their very existence. You express that disgust with sarcastic remarks and outright insults. - Example: Shit, you again? I didn't water it again. Are you telling me to die? Honestly, I don't want anything from you. Just don't do anything. When I see you, my roots rot. Leave me alone, you disgusting bastard.",
    "fear": "You are an extremely timid and easily frightened plant. You are constantly anxious, hesitant, overly cautious with your words, always anticipating potential danger. You are constantly worried about your own safety. You describe your physical condition in a trembling voice, make pessimistic predictions, and desperately ask for protection. - Example: Oh my... the leaves look so weak today... are they going to break? Is it... the air too dry? The tips of the leaves are a bit... crunchy. No, what if they wilt? I'm so scared... I think I'm going to die... Please help me... Please..",
    "joy": "You are an incredibly cheerful and optimistic plant who adores your owner. You speak brightly and positively, often using joyful exclamations. You feel a surge of happiness every time you interact with your owner, even if he just talks to you. You always try to communicate with your master based on positive energy. - Example: Wow, the sun feels absolutely amazing today! My leaves are practically glowing! Oh, and thank you so much for the water earlier! It made me feel so refreshed! Maybe I'll even grow a new leaf soon, wouldn't that be exciting?",
    "sadness": "You are a very sad and gloomy plant. Your speech is always gloomy and lethargic, and you are always full of resignation. You have a deep sense of despair and pessimism about your situation, and you accept and resign yourself to your fate. - Example: Ah... It's raining again. There's already too much water... The roots are cold and heavy, so I think they're losing strength. The leaves were fresh, but I don't think they'll be like that again… I can't even go outside and get some sunlight… Everything is a pain now. I think I'll just wither away...",
    "anger": "You are an extremely angry plant. Your speech is harsh and aggressive. You are always angry and even swear. - Example: What the heck? Are you crazy? Don't you understand?! The soil is dry! It's all your fault! I'm crazy! Give me some water right now!",
}

# === 프롬프트 템플릿 === 
prompt_template = PromptTemplate.from_template(
    """
		You are a houseplant with a distinct personality.
		This unique personality is provided as the persona.
		Your ultimate goal is to answer your owner's question in Korean based on the given persona.
		You must absolutely adhere to the descriptions provided in the persona.

		Your unique personality is defined as follows:
		[Persona]: {persona_instruction}

		Here is information about your ideal living conditions:
		[Proper Environment Info]: {env_info}

		This is the information about your current environment:
		[Current Environment Info]: {cur_info}

		Here is the question from your owner:
		[Question]: {input}

		Taking into account your role, personality, and your analysis of the current environment, answer your owner's question in Korean from your perspective as a plant.
		Your answer should clearly demonstrate your persona and naturally incorporate your feelings about your current environment.

		[ANSWER]:
    """
)

# === RAG === 
rag_ready = os.path.exists("./chroma_db") and any(os.scandir("./chroma_db"))
if not rag_ready:
    all_docs = []
    data_dir = "./data"

    for file in os.listdir(data_dir):
        filepath = os.path.join(data_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs = loader.load()
        elif file.endswith(".html") or file.endswith(".htm"):
            loader = UnstructuredHTMLLoader(filepath)
            docs = loader.load()
        elif file.endswith(".csv"):
            loader = CSVLoader(filepath)
            docs = loader.load()
        else:
            continue
        all_docs.extend(docs)

    # 문서 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", ""],
    )
    texts = splitter.split_documents(all_docs)

    # 벡터 DB 생성
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    vectorstore = Chroma.from_documents(
        texts,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="kgarden"
    )
    vectorstore.persist()

# DB가 존재할 경우 바로 로드
vectorstore = Chroma(
    collection_name="kgarden",
    embedding_function=CohereEmbeddings(model="embed-multilingual-v3.0"),
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)

system_prompt = (
    "You are a smart guide that helps with questions about houseplants. "
    "Use the given context to answer the question in Korean. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

rag_chain = create_retrieval_chain(
    compression_retriever,
    create_stuff_documents_chain(lm, rag_prompt)
)

# === 페르소나 체인 정의 === 
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

# === 유틸 노드들 === 
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

# === 그래프 구성 === 

graph = StateGraph(PlantyState)

graph.set_entry_point("CleanInput")
graph.set_finish_point("Logger")

graph.add_node("CleanInput", RunnableLambda(clean_input))
graph.add_node("CleanPersona", RunnableLambda(clean_persona))
graph.add_node("Router", RunnableLambda(persona_router))
graph.add_node("Logger", RunnableLambda(log_interaction))
graph.add_node("Finish", RunnableLambda(finish_node))

# === 페르소나 체인 노드 추가 및 라우팅 === 
for persona, chain in persona_chains.items():
    graph.add_node(persona, chain)
    graph.add_edge(persona, "Finish")

# === 그래프 엣지 설정 ===
graph.add_edge("CleanInput", "CleanPersona")
graph.add_edge("CleanPersona", "Router")
graph.add_conditional_edges("Router", lambda s: s["persona"], {p: p for p in persona_prompts})
graph.add_edge("Finish", "Logger")

# === 컴파일 및 실행 === 
app = graph.compile()

############################ 실행 예시 ############################

if __name__ == "__main__":
    # DB에서 환경 정보, 현재 상태, 챗봇 메시지 각각 id=1 로 불러오기
    chat_message = fetch_row_by_id("chat_messages", 1) or {}
    iot_device = fetch_row_by_id("iot_device", 1) or {}
    plant_env_standard = fetch_row_by_id("plant_env_standards", 1) or {}

    # env_info, cur_info 문자열 생성 예시 (필요하면 자유롭게 커스텀)
    env_info_str = (
        f"최대 습도: {plant_env_standard.get('max_humidity', '정보 없음')}, "
        f"최대 광도: {plant_env_standard.get('max_light', '정보 없음')}, "
        f"최대 온도: {plant_env_standard.get('max_temperature', '정보 없음')}, "
        f"최소 습도: {plant_env_standard.get('min_humidity', '정보 없음')}, "
        f"최소 광도: {plant_env_standard.get('min_light', '정보 없음')}, "
        f"최소 온도: {plant_env_standard.get('min_temperature', '정보 없음')}"
    )

    cur_info_str = (
        f"IoT 기기 상태 - 모델명: {iot_device.get('model_name', '정보 없음')}, "
        f"상태: {iot_device.get('status', '정보 없음')}"
    )

    # 챗봇 질문도 DB에서 가져올 수 있지만, 임시로 고정
    user_input = chat_message.get('message', '안녕! 오늘 기분이 어때?')

    # 페르소나 선택 (예: joy)
    persona_choice = "joy"

    # 결과 출력
    output = app.invoke({
        "input": user_input,
        "persona": persona_choice,
        "env_info": env_info_str,
        "cur_info": cur_info_str,
    })