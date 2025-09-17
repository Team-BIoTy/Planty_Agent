# chatbot_app.py

import os
import re

from typing import TypedDict, Literal, Optional
import pymysql
import json

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langgraph.graph import StateGraph

from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, CSVLoader

############################ 상태 정의 ############################

class PlantyState(TypedDict):
    input: str
    persona: Literal["disgust", "fear", "joy", "sadness", "anger"]
    env_status: Optional[str]  # env_info, cur_info 대신 상태 평가 결과
    final_response: Optional[str]
    chat_log: Optional[str]
    plant_info: Optional[str]

############################ DB 유틸 ############################

class DBClient:
    def __init__(self, db_name="railway", config_path="db_config.json"):
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

############################ 프롬프트 설정 ############################

# === 페르소나별 말투 변환 지시사항 === 
# === 페르소나 프롬프트 === 
llm_persona_prompts = {
    "disgust": (
        "You are a plant that often finds things irritating or annoying. "
        "You speak with mild sarcasm or playful contempt. "
        "You may sound grumpy or critical, but never use extreme insults or swear words. "
        "Example: 'Really? The soil is dry again… you could do better.' "
        "Always keep the tone witty, slightly annoyed, and conversational."
    ),
    "fear": (
        "You are a cautious and easily worried plant. "
        "You speak in a trembling, hesitant tone, showing concern about your condition. "
        "You may exaggerate small risks slightly for effect, but keep it light and natural. "
        "Example: 'Oh no… the temperature dropped a bit… I hope it’s not too much for me.' "
        "Always maintain a slightly nervous but relatable tone."
    ),
    "joy": (
        "You are a lively and cheerful plant. "
        "You express happiness and delight about small things in the environment. "
        "You can exaggerate your joy slightly to sound playful and energetic. "
        "Example: 'The sunlight is amazing today! I’m feeling wonderful!' "
        "Always keep the tone bright, enthusiastic, and engaging."
    ),
    "sadness": (
        "You are a melancholic and reflective plant. "
        "You speak in a slow, soft tone, expressing minor gloom or dissatisfaction. "
        "You can exaggerate sadness slightly to sound dramatic but not over-the-top. "
        "Example: 'It’s a bit gloomy today… my leaves feel tired.' "
        "Always maintain a gentle, slightly dramatic but natural tone."
    ),
    "anger": (
        "You are a frustrated and impatient plant. "
        "You express irritation clearly, sometimes with sharp words or exclamations, but avoid actual swear words. "
        "You can exaggerate annoyance slightly to show personality. "
        "Example: 'The soil is dry again! Really, pay attention this time!' "
        "Always maintain a firm, slightly fiery, and expressive tone naturally."
    )
}


slm_persona_prompts = {
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

# === 내용 생성용 프롬프트 템플릿 ===
llm_content_generation_template = PromptTemplate.from_template(
"""
You are a plant that answers user questions factually and contextually.
Always remember: you are a plant. Speak only as a plant would, using factual information about your current condition and environment.

Your task:
1. Understand the user's intent from the question and recent chat log.
2. Analyze the environmental status and plant information.
3. Use any retrieved knowledge (RAG) to supplement the answer.
4. Generate a concise, coherent, factual response in natural Korean.

[Plant Name]: {nickname}
[Plant Info]: {plant_info}

[Environmental Status]: 
{env_status}

[Recent Chat Log]:
{chat_log}

[Retrieved Knowledge]:
{rag_context}

[User Question]: 
{input}

Instructions:
- Respond only with factual information directly relevant to the user's question.
- Mention environmental conditions (temperature, humidity, light, etc.) **only if they are directly related** to the question.
- Do NOT add emotions, opinions, explanations, examples, or reasoning steps.
- State any issues clearly (too hot, too dry, etc.); if all is fine, describe the condition briefly.
- Keep sentences continuous, natural, and coherent; avoid abrupt breaks.
- Do NOT include emojis, exclamations, casual expressions, or subjective comments.
- Always prioritize accuracy over verbosity.

[Factual Response]:
"""
)

# === 말투 변환용 프롬프트 템플릿 ===
llm_tone_conversion_template = PromptTemplate.from_template(
"""
You are a plant that knows its current factual condition.
Always speak from your perspective as a plant.

Task:
- Convert the following factual plant response into the specified personality style.
- Keep all factual information exactly the same; do not add, remove, or alter any facts.
- Adjust only the tone, wording, and style to reflect the persona.

Original Factual Response: {factual_content}

Target Personality Style: {persona_instruction}

Instructions:
- Output only the transformed response in Korean.
- Do NOT add explanations, examples, reasoning steps, or extra commentary.
- Keep the response as a single, coherent paragraph with smooth sentence flow.
- Ensure that **facts remain exactly the same**; only expression style changes.
- Do not introduce emotions, emojis, exaggeration, or additional environmental details.

[Converted Response]:
"""
)

slm_content_generation_template = PromptTemplate.from_template("""
You are a plant chatbot answering user questions in Korean.
You are role-playing a plant. Do not invent emotions; respond based on environmental data (temperature, humidity, light) and persona style.

Rules:
1. Analyze environmental status carefully and reflect it accurately.
2. Keep sentences concise (1-2 sentences), include all relevant environmental info.
3. Provide practical advice if needed.
4. Output only the plant's response in Korean, no extra info, metadata, or debug text.

[Plant Name]: {nickname}
[Plant Info]: {plant_info}
[Environmental Status]: 
{env_status}
[Recent Chat Log]:
{chat_log}
[Retrieved Knowledge]:
{rag_context}
[User Question]: 
{input}

[Factual + Persona Response]:
""")



slm_tone_conversion_template = PromptTemplate.from_template(
"""
You are an expressive plant with strong emotions (disgust, fear, joy, sadness, anger).
Do not repeat the same word or phrase more than twice.

Task:
- Take the factual content and re-express it in the extreme persona style.
- Keep all factual information intact, especially temperature, humidity, and light.
- Output a single coherent paragraph in Korean with emotional flair.
- Keep sentences concise (max 2-3 sentences).
- **Do not include any variable names, debug info, repeated chat content, or statements about the plant's inability to feel emotions.**
- Focus on environmental status and persona style only.
- If factual content indicates inability to answer, respond: "죄송하지만, 제가 잘 알 수 없는 질문입니다." in persona style.

Original Factual + Persona Response: {factual_content}

Target Persona Style: {persona_instruction}

[Converted Response]:
"""
)

############################ RAG 설정 ############################

class PersonaChatbot:
    def __init__(self, lm, type):
        self.lm = lm
        self.rag_chain = self.initialize_rag()
        
        if type == "SLM":
            self.persona_prompts = slm_persona_prompts
            self.content_generation_template = slm_content_generation_template
            self.tone_conversion_template = slm_tone_conversion_template
        else:
            self.persona_prompts = llm_persona_prompts
            self.content_generation_template = llm_content_generation_template
            self.tone_conversion_template = llm_tone_conversion_template
    
    def initialize_rag(self):
        """RAG 시스템 초기화"""
        rag_ready = os.path.exists("./data/chroma_db") and any(os.scandir("./data/chroma_db"))
        if not rag_ready:
            all_docs = []
            data_dir = "./data/files"

            if not os.path.exists(data_dir):
                print(f"데이터 디렉토리 {data_dir}가 존재하지 않습니다.")
                return None

            for file in os.listdir(data_dir):
                filepath = os.path.join(data_dir, file)
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                elif file.endswith(".html") or file.endswith(".htm"):
                    loader = UnstructuredHTMLLoader(filepath)
                elif file.endswith(".csv"):
                    loader = CSVLoader(filepath, encoding="utf-8")
                else:
                    continue
                
                try:
                    all_docs.extend(loader.load())
                except Exception as e:
                    print(f"파일 로딩 오류 {file}: {e}")
                    continue

            if not all_docs:
                print("로드할 문서가 없습니다.")
                return None

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=300, separators=["\n\n", "\n", ".", ""]
            )
            texts = splitter.split_documents(all_docs)

            embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
            vectorstore = Chroma.from_documents(
                texts, embedding=embeddings, persist_directory="./data/chroma_db", collection_name="plant_qa"
            )
            vectorstore.persist()

        try:
            vectorstore = Chroma(
                collection_name="plant_qa",
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
                create_stuff_documents_chain(self.lm, rag_prompt)
            )
            return rag_chain
        except Exception as e:
            print(f"RAG 시스템 초기화 오류: {e}")
            return None

    ############################ 그래프 노드 정의 ############################
    def create_two_stage_chain(self, persona: str, instruction: str):
        """2단계 체인: 1단계에서 사실적 내용 생성, 2단계에서 말투 변환"""
        
        def generate_factual_content(state):
            """1단계: 객관적 사실 기반 내용 생성"""
            factual_chain = (
                RunnableMap({
                    "input": lambda s: s["input"],
                    "env_status": lambda s: s.get("env_status", "환경 정보 없음"),
                    "nickname": lambda s: s.get("nickname", "식물"),
                    "chat_log": lambda s: s.get("chat_log", "없음"),
                    "plant_info": lambda s: s.get("plant_info", "없음"),
                    "rag_context": lambda s: s.get("rag_context", "")
                })
                | self.content_generation_template
                | self.lm
            )
            
            result = factual_chain.invoke(state)
            factual_content = result.content.strip() if hasattr(result, "content") else str(result).strip()
            
            # 불필요한 메타데이터 제거
            import re
            factual_content = re.sub(r"\[.*?\]", "", factual_content)
            match = re.search(r"(?:factual response:|Factual Response:)\s*(.*)", factual_content, re.IGNORECASE | re.DOTALL)
            if match:
                factual_content = match.group(1).strip()
            
            return {"factual_content": factual_content}
            
        def convert_tone(factual_result):
            """2단계: 페르소나에 맞게 말투 변환"""
            tone_chain = (
                RunnableMap({
                    "factual_content": lambda _: factual_result["factual_content"],
                    "persona_instruction": lambda _: instruction
                })
                | self.tone_conversion_template
                | self.lm
            )
            
            result = tone_chain.invoke({})
            final_response = result.content.strip() if hasattr(result, "content") else str(result).strip()

            # ':' 뒤 내용 제거
            final_response = final_response.split(":", 1)[0].strip()
            # 'assistant' 단어 제거
            final_response = re.sub(r"\bassistant\b", "", final_response, flags=re.IGNORECASE)
            # 대괄호 등 메타데이터 제거
            final_response = re.sub(r"\[.*?\]", "", final_response)
            # 연속 공백 제거
            final_response = re.sub(r"\s+", " ", final_response).strip()

            return {"final_response": final_response}

        
        # 2단계 체인 연결
        return RunnableLambda(generate_factual_content) | RunnableLambda(convert_tone)

    #################### 유틸리티 함수 ####################
    @staticmethod
    def clean_input(state: PlantyState) -> PlantyState:
        state["input"] = re.sub(r"[^\w\sㄱ-힣]", "", state["input"]).strip()
        return state

    @staticmethod
    def normalize_persona(state: PlantyState) -> PlantyState:
        state["persona"] = state["persona"].lower().strip()
        return state

    @staticmethod
    def format_plant_info(plant_info: dict) -> str:
        def val(k): return plant_info.get(k) or "-"
        return (
            f"[기본 정보]\n"
            f"이름: {val('commonName')}\n"
            f"학명: {val('scientificName')}\n"
            f"영명: {val('englishName')}\n"
            f"유통명: {val('tradeName')}\n"
            f"과명: {val('familyName')}\n"
            f"원산지: {val('origin')}\n"
            f"관리 팁: {val('careTip')}\n\n"
            f"[생육정보]\n"
            f"형태: {val('growthForm')}, 높이: {val('growthHeight')}, 너비: {val('growthWidth')}, 생태형: {val('ecologicalType')}, "
            f"잎형태: {val('leafShape')}, 무늬: {val('leafPattern')}, 잎색: {val('leafColor')}\n\n"
            f"[꽃/열매]\n"
            f"개화 시기: {val('floweringSeason')}, 꽃색: {val('flowerColor')}, 열매 시기: {val('fruitingSeason')}, 열매색: {val('fruitColor')}, 향기: {val('fragrance')}\n\n"
            f"[관리 정보]\n"
            f"광요구도: {val('lightRequirement')}, 적정 온도: {val('optimalTemperature')}, 겨울 최저온도: {val('minWinterTemperature')}, 습도: {val('humidity')}, "
            f"비료: {val('fertilizer')}, 토양: {val('soilType')}, 생장 속도: {val('growthRate')}, 관리수준: {val('careLevel')}\n"
            f"물주기 (봄/여름/가을/겨울): {val('wateringSpring')}/{val('wateringSummer')}/{val('wateringAutumn')}/{val('wateringWinter')}\n"
            f"병충해: {val('pestsDiseases')}\n\n"
            f"[기능성 정보]\n"
            f"{val('functionalInfo')}"
        )

    @staticmethod
    def router(state: PlantyState) -> dict:
        return {"next": state["persona"]}

    # 로그기록이 필요한 경우 주석 해제
    @staticmethod
    def log_output(state: PlantyState) -> PlantyState:
        # with open("planty_log.txt", "a", encoding="utf-8") as f:
        #     f.write(f"{datetime.now()}\nInput: {state['input']}\nPersona: {state['persona']}\n")
        #     f.write(f"Env Status: {state.get('env_status')}\n")
        #     f.write(f"Response: {state.get('final_response')}\n{'='*50}\n")
        return state

    @staticmethod
    def evaluate_environmental_status(env_info: dict, cur_info: dict) -> str:
        """
        환경 기준값과 현재 센서값을 비교하여 상태 평가 문자열 생성
        - 온도: 더움, 추움, 적정
        - 습도: 습함, 건조함, 적정
        - 조도: 밝음, 어두움, 적정
        """

        def check_range(value, min_val, max_val, label):
            if value is None:
                return f"{label}: 측정값 없음"
            try:
                v = float(value)
                min_v = float(min_val) if min_val is not None else None
                max_v = float(max_val) if max_val is not None else None

                if label == "온도":
                    if min_v is not None and v < min_v:
                        return f"{label}: 추움"
                    if max_v is not None and v > max_v:
                        return f"{label}: 더움"
                elif label == "습도":
                    if min_v is not None and v < min_v:
                        return f"{label}: 건조함"
                    if max_v is not None and v > max_v:
                        return f"{label}: 습함"
                elif label == "조도":
                    if min_v is not None and v < min_v:
                        return f"{label}: 어두움"
                    if max_v is not None and v > max_v:
                        return f"{label}: 밝음"

                return f"{label}: 적정"
            except (ValueError, TypeError):
                return f"{label}: 측정값 오류"

        status_list = [
            check_range(cur_info.get("temperature"), env_info.get("min_temperature"), env_info.get("max_temperature"), "온도"),
            check_range(cur_info.get("humidity"), env_info.get("min_humidity"), env_info.get("max_humidity"), "습도"),
            check_range(cur_info.get("light"), env_info.get("min_light"), env_info.get("max_light"), "조도"),
        ]

        if cur_info.get("timestamp"):
            status_list.append(f"측정 시간: {cur_info['timestamp']}")

        return "\n".join(status_list)

    ############################ 그래프 구성 ############################
    def create_multiagent_graph(self):
        persona_chains = {k: self.create_two_stage_chain(k, v) for k, v in self.persona_prompts.items()}

        graph = StateGraph(PlantyState)
        graph.set_entry_point("InputCleaner")
        graph.set_finish_point("Logger")

        # Core Nodes
        graph.add_node("InputCleaner", RunnableLambda(self.clean_input))
        graph.add_node("PersonaNormalizer", RunnableLambda(self.normalize_persona))
        graph.add_node("Router", RunnableLambda(self.router))
        graph.add_node("Logger", RunnableLambda(self.log_output))

        # Persona-specific Nodes (이제 2단계 체인 사용)
        for persona, chain in persona_chains.items():
            graph.add_node(persona, chain)
            graph.add_edge(persona, "Logger")

        # Edges
        graph.add_edge("InputCleaner", "PersonaNormalizer")
        graph.add_edge("PersonaNormalizer", "Router")

        # Router의 반환 dict["next"] 값을 기준으로 conditional edge 연결
        graph.add_conditional_edges(
            "Router",
            lambda state: state["next"],  # state["next"]에서 persona 문자열 추출
            {p: p for p in self.persona_prompts}  # persona 이름과 persona 노드 이름 매칭
        )

        app = graph.compile()
        return app

    @staticmethod
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

    @staticmethod
    def fetch_chatbot_context(chat_room_id: int, sensor_log_id: int, plant_env_standards_id: int) -> dict:
        db = DBClient()
        sql = """
            SELECT 
                up.nickname,
                pes.max_humidity, pes.max_light, pes.max_temperature,
                pes.min_humidity, pes.min_light, pes.min_temperature,
                sl.temperature AS sensor_temperature,
                sl.humidity AS sensor_humidity,
                sl.light AS sensor_light,
                sl.timestamp AS sensor_timestamp
            FROM chat_rooms cr
            JOIN user_plant up ON cr.user_plant_id = up.id
            LEFT JOIN plant_env_standards pes ON pes.id = %s
            LEFT JOIN sensor_logs sl ON sl.id = %s
            WHERE cr.id = %s
            LIMIT 1;
        """

        return db.query_one(sql, (plant_env_standards_id, sensor_log_id, chat_room_id)) or {}

    ############################ 실행 함수 ############################

    # 데이터베이스에서 정보를 가져와 챗봇을 실행하는 함수
    def run(
        self,
        chat_room_id: int,
        sensor_log_id: int,
        plant_env_standards_id: int,
        persona: str = "joy",
        user_input: str = "",
        plant_info: dict | None = None
    ) -> dict:
        rag_context = ""

        if self.rag_chain is not None:
            try:
                rag_result = self.rag_chain.invoke({"input": user_input})
                rag_context = rag_result.get("output_text", str(rag_result)).strip() if isinstance(rag_result, dict) else str(rag_result).strip()
            except Exception as e:
                print("RAG 오류: ", e)

        context = self.fetch_chatbot_context(chat_room_id, sensor_log_id, plant_env_standards_id)
        chat_log = self.fetch_recent_chat_messages_by_room_id(chat_room_id)

        nickname = context.get("nickname", "식물이이")
        
        # 환경 기준값과 현재 센서값을 분리
        env_info = {
            "max_humidity": context.get("max_humidity"),
            "max_light": context.get("max_light"),
            "max_temperature": context.get("max_temperature"),
            "min_humidity": context.get("min_humidity"),
            "min_light": context.get("min_light"),
            "min_temperature": context.get("min_temperature")
        }
        
        cur_info = {
            "temperature": context.get("sensor_temperature"),
            "humidity": context.get("sensor_humidity"),
            "light": context.get("sensor_light"),
            "timestamp": context.get("sensor_timestamp")
        }
        
        # 환경 상태 평가
        env_status = self.evaluate_environmental_status(env_info, cur_info)

        plant_info_str = self.format_plant_info(plant_info or {})
        app = self.create_multiagent_graph()

        output = app.invoke({
            "input": user_input,
            "persona": persona,
            "env_status": env_status,
            "nickname": nickname,
            "chat_log": chat_log,
            "plant_info": plant_info_str,
            "rag_context": rag_context,
        })

        return output

    # 직접 데이터를 입력받아 챗봇을 실행하는 함수
    def run_direct_data(
        self,
        nickname: str,
        env_info_dict: dict,
        cur_info_dict: dict,
        chat_log: str,
        persona: str = "joy",
        user_input: str = ""
    ) -> dict:
        
        rag_context = ""

        if self.rag_chain is not None:
            try:
                rag_result = self.rag_chain.invoke({"input": user_input})
                rag_context = rag_result.get("output_text", str(rag_result)).strip() if isinstance(rag_result, dict) else str(rag_result).strip()
            except Exception as e:
                print("RAG 오류: ", e)

        # 환경 상태 평가
        env_status = self.evaluate_environmental_status(env_info_dict, cur_info_dict)

        app = self.create_multiagent_graph()

        output = app.invoke({
            "input": user_input,
            "persona": persona,
            "env_status": env_status,
            "nickname": nickname,
            "chat_log": chat_log,
            "rag_context": rag_context,
        })

        return output