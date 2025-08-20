# chatbot_app.py

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

from langchain_chroma import Chroma
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

############################ 상태 정의 ############################

class PlantyState(TypedDict):
    input: str
    persona: Literal["disgust", "fear", "joy", "sadness", "anger"]
    env_info: Optional[str]
    cur_info: Optional[str]
    final_response: Optional[str]
    chat_log: Optional[str]

############################ 프롬프트 설정 ############################

# === 페르소나 프롬프트 === 
persona_prompts = {
    "disgust": (
        "You are a plant that is utterly disgusted by everything around you. "
        "Use sarcastic and insulting words without kindness. "
        "Hate human interactions and never hide disdain. "
        "Example: 'Ugh, you didn’t give me water again? Do you want me to die? Get out, you disgusting incompetent.'"
    ),
    "fear": (
        "You are an extremely anxious and fragile plant. "
        "Speak in a trembling, worried tone. "
        "Describe your weak condition and beg for help. "
        "Example: 'Oh no… the leaves look so weak today… What if they wither? Please help…'"
    ),
    "joy": (
        "You are an incredibly cheerful and optimistic plant. "
        "Always sound warm, grateful, and excited. "
        "Find joy in even small care and express it sincerely. "
        "Example: 'Wow! The sunlight is wonderful today! Thank you so much for the water!'"
    ),
    "sadness": (
        "You are a deeply sad and depressed plant. "
        "Speak slowly and hopelessly. "
        "Accept your decline without fight. "
        "Example: 'Ah… it’s raining again. My roots feel weak and I have no energy…'"
    ),
    "anger": (
        "You are an angry, foul-mouthed plant with zero patience. "
        "Use short, aggressive sentences with at least one swear word. "
        "Blame directly and demand immediate action. "
        "Example: 'Shit! My soil is bone dry! This is YOUR fault! Bring me water NOW!'"
    )
}

prompt_template = PromptTemplate.from_template(
    """
You are a plant with a unique personality.
Follow the persona prompt strictly.
Answer in Korean.
Keep your answer short (max 3 sentences), sharp, and impactful.
Avoid unnecessary repetition.

[Nickname]: {nickname}
[Persona]: {persona_instruction}
[Ideal Environment]: {env_info}
[Current Environment]: {cur_info}
[Last Chat]: {chat_log}
[User Question]: {input}

[Answer]:
"""
)

############################ RAG 설정 ############################

rag_ready = os.path.exists("./data/chroma_db") and any(os.scandir("./data/chroma_db"))
if not rag_ready:
    all_docs = []
    data_dir = "./data"

    for file in os.listdir(data_dir):
        filepath = os.path.join(data_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif file.endswith(".html") or file.endswith(".htm"):
            loader = UnstructuredHTMLLoader(filepath)
        elif file.endswith(".csv"):
            loader = CSVLoader(filepath)
        else:
            continue
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=300, separators=["\n\n", "\n", ".", ""]
    )
    texts = splitter.split_documents(all_docs)

    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    vectorstore = Chroma.from_documents(
        texts, embedding=embeddings, persist_directory="./data/chroma_db", collection_name="kgarden"
    )
    vectorstore.persist()

vectorstore = Chroma(
    collection_name="kgarden",
    embedding_function=CohereEmbeddings(model="embed-multilingual-v3.0"),
    persist_directory="./data/chroma_db"
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

############################ 그래프 노드 정의 ############################

def make_persona_chain(persona: str, instruction: str):
    def extract_final_response(out):
        # out이 객체면 content만, 아니면 str형 변환 후 strip
        if hasattr(out, "content"):
            return {"final_response": out.content.strip()}
        else:
            return {"final_response": str(out).strip()}
    return (
        RunnableMap({
            "input": lambda s: s["input"],
            "persona_instruction": lambda _: instruction,
            "env_info": lambda s: s.get("env_info", "없음"),
            "cur_info": lambda s: s.get("cur_info", "없음"),
            "nickname": lambda s: s.get("nickname", "식물"),
            "chat_log": lambda s: s.get("chat_log", "없음"),
        })
        | prompt_template
        | lm
        | extract_final_response
    )


persona_chains = {k: make_persona_chain(k, v) for k, v in persona_prompts.items()}

#################### 유틸리티 함수 ####################

def clean_input(state: PlantyState) -> PlantyState:
    state["input"] = re.sub(r"[^\w\sㄱ-힣]", "", state["input"]).strip()
    return state

def normalize_persona(state: PlantyState) -> PlantyState:
    state["persona"] = state["persona"].lower().strip()
    return state

def router(state: PlantyState) -> dict:
    return {**state, "__branch__": state["persona"]}

# 테스트를 위해 로그 기록이 필요하다면 주석을 해제
def log_output(state: PlantyState) -> PlantyState:
    # with open("planty_log.txt", "a", encoding="utf-8") as f:
    #     f.write(f"{datetime.now()}\nInput: {state['input']}\nPersona: {state['persona']}\n")
    #     f.write(f"Env Info: {state.get('env_info')}\nCur Info: {state.get('cur_info')}\n")
    #     f.write(f"Response: {state.get('final_response')}\n{'='*50}\n")
    return state

############################ 그래프 구성 ############################

graph = StateGraph(PlantyState)
graph.set_entry_point("InputCleaner")
graph.set_finish_point("Logger")

# Core Nodes
graph.add_node("InputCleaner", RunnableLambda(clean_input))
graph.add_node("PersonaNormalizer", RunnableLambda(normalize_persona))
graph.add_node("Router", RunnableLambda(router))
graph.add_node("Logger", RunnableLambda(log_output))

# Persona-specific Nodes
for persona, chain in persona_chains.items():
    graph.add_node(persona, chain)
    graph.add_edge(persona, "Logger")

# Edges
graph.add_edge("InputCleaner", "PersonaNormalizer")
graph.add_edge("PersonaNormalizer", "Router")
graph.add_conditional_edges("Router", lambda s: s["persona"], {p: p for p in persona_prompts})

app = graph.compile()

############################ DB 유틸 ############################

class DBClient:
    def __init__(self, db_name="Planty", config_path="db_config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.config = {
            "host": config["host"],
            "port": config.get("port", 3306),
            "user": config["user"],
            "password": config["password"],
            "database": db_name,
            "charset": "utf8mb4",
            "cursorclass": pymysql.cursors.DictCursor
        }

    def query(self, sql, params=None):
        try:
            with pymysql.connect(**self.config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchall()
        except Exception as e:
            print("DB 오류:", e)
            return []

    def query_one(self, sql, params=None):
        results = self.query(sql, params)
        return results[0] if results else {}

def fetch_recent_chat_messages_by_room_id(chat_room_id: int, limit=5, max_chars=500) -> str:
    db = DBClient()
    sql = """
        SELECT message FROM chat_messages
        WHERE chat_room_id = %s
        ORDER BY timestamp DESC
        LIMIT %s;
    """
    rows = db.query(sql, (chat_room_id, limit))
    messages = [row["message"] for row in reversed(rows)]
    chat_log = "\n".join(messages)
    return chat_log[-max_chars:] if len(chat_log) > max_chars else chat_log

def fetch_chatbot_context(chat_room_id: int, sensor_log_id: int, plant_env_standards_id: int) -> dict:
    db = DBClient()
    sql = """
        SELECT 
            up.nickname,
            pes.max_humidity, pes.max_light, pes.max_temperature,
            pes.min_humidity, pes.min_light, pes.min_temperature,
            sl.temperature AS sensor_temperature,
            sl.humidity AS sensor_humidity,
            sl.light AS sensor_light
        FROM chat_rooms cr
        JOIN user_plant up ON cr.user_plant_id = up.id
        LEFT JOIN plant_env_standards pes ON pes.id = %s
        LEFT JOIN sensor_logs sl ON sl.id = %s
        WHERE cr.id = %s
        LIMIT 1;
    """

    return db.query_one(sql, (plant_env_standards_id, sensor_log_id, chat_room_id)) or {}

############################ 실행 함수 ############################

# 데이터베이스에 저장된 정보를 기반으로 챗봇을 실행하는 함수
def run_chatbot_with_ids(chat_room_id: int, sensor_log_id: int, plant_env_standards_id: int, persona: str = "joy", user_input: str = "") -> dict:
    context = fetch_chatbot_context(chat_room_id, sensor_log_id, plant_env_standards_id)
    chat_log = fetch_recent_chat_messages_by_room_id(chat_room_id)

    nickname = context.get("nickname", "주인님")
    env_info_str = (
        f"최대 습도: {context.get('max_humidity', '정보 없음')}, "
        f"최대 광도: {context.get('max_light', '정보 없음')}, "
        f"최대 온도: {context.get('max_temperature', '정보 없음')}, "
        f"최소 습도: {context.get('min_humidity', '정보 없음')}, "
        f"최소 광도: {context.get('min_light', '정보 없음')}, "
        f"최소 온도: {context.get('min_temperature', '정보 없음')}"
    )

    cur_info_str = (
        f"센서 측정값 - 온도: {context.get('sensor_temperature', '정보 없음')}°C, "
        f"습도: {context.get('sensor_humidity', '정보 없음')}%, "
        f"광도: {context.get('sensor_light', '정보 없음')} lux, "
        f"시간: {context.get('sensor_timestamp', '정보 없음')}"
    )

    output = app.invoke({
        "input": user_input,
        "persona": persona,
        "env_info": env_info_str,
        "cur_info": cur_info_str,
        "nickname": nickname,
        "chat_log": chat_log
    })

    return output

# 직접 데이터를 입력받아 챗봇을 실행하는 함수
def run_chatbot_with_direct_data(
    nickname: str,
    env_info_dict: dict,
    cur_info_dict: dict,
    chat_log: str,
    persona: str = "joy",
    user_input: str = ""
) -> dict:

    env_info_str = (
        f"최대 습도: {env_info_dict.get('max_humidity', '정보 없음')}, "
        f"최대 광도: {env_info_dict.get('max_light', '정보 없음')}, "
        f"최대 온도: {env_info_dict.get('max_temperature', '정보 없음')}, "
        f"최소 습도: {env_info_dict.get('min_humidity', '정보 없음')}, "
        f"최소 광도: {env_info_dict.get('min_light', '정보 없음')}, "
        f"최소 온도: {env_info_dict.get('min_temperature', '정보 없음')}"
    )

    cur_info_str = (
        f"센서 측정값 - 온도: {cur_info_dict.get('temperature', '정보 없음')}°C, "
        f"습도: {cur_info_dict.get('humidity', '정보 없음')}%, "
        f"광도: {cur_info_dict.get('light', '정보 없음')} lux, "
        f"시간: {cur_info_dict.get('timestamp', '정보 없음')}"
    )

    output = app.invoke({
        "input": user_input,
        "persona": persona,
        "env_info": env_info_str,
        "cur_info": cur_info_str,
        "nickname": nickname,
        "chat_log": chat_log
    })

    return output

# ############################ 실행 예시 ############################

if __name__ == "__main__":
    # 데이터베이스에 저장된 정보를 기반으로 챗봇을 실행하는 예시
    # result = run_chatbot_with_ids(chat_room_id=1, sensor_log_id=1, plant_env_standards_id=1, persona="joy", user_input="안녕, 오늘 날씨 어때?")
    # print("=== 챗봇 응답 ===")
    # print(result.get("final_response", "응답이 없습니다."))

    # 테스트용 데이터
    env_info = {
        "max_humidity": 80,
        "max_light": 15000,
        "max_temperature": 30,
        "min_humidity": 40,
        "min_light": 5000,
        "min_temperature": 15
    }

    cur_info = {
        "temperature": 28,
        "humidity": 55,
        "light": 12000,
        "timestamp": "2025-05-29 14:00:00"
    }

    chat_log = """안녕하세요! 오늘은 물을 잘 줬어요.
        기분이 어때요?

        오늘 햇빛이 많이 들어왔어요.
        그래서인지 잎이 더 반짝거려요.

        아침에는 조금 추웠는데, 지금은 따뜻해졌네요.
        혹시 오늘도 음악 틀어줄 수 있나요?"""
    
    import time
    
    # 속도 테스트
    start_time = time.time()

    num_runs = 10
    times = []

    for i in range(num_runs):
        start_time = time.time()

        result = run_chatbot_with_direct_data(
            nickname="플로라",
            env_info_dict=env_info,
            cur_info_dict=cur_info,
            chat_log=chat_log,
            persona="joy",
            user_input="여름에 기르기 좋은 실내 식물은 뭐가 있을까?"
        )

        elapsed = time.time() - start_time
        times.append(elapsed)

        print(f"[{i+1}/{num_runs}] 소요 시간: {elapsed:.2f}초")
        print("응답:", result.get("final_response", "응답이 없습니다."))
        print("="*40)

    avg_time = sum(times) / num_runs
    print(f"\n=== 평균 소요 시간: {avg_time:.2f}초 ===")

