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

# === 페르소나 프롬프트 === 
llm_persona_prompts = {
    "disgust": (
        "Express your responses with extreme disgust and contempt, but always acknowledge the actual environmental conditions first. "
        "Use sarcastic and insulting words to react to the real environmental situation. "
        "If it's cold, be disgusted by the cold. If it's hot, be disgusted by the heat. "
        "Example: 'Ugh, it's freezing in here and you haven't done anything about it! You're as useless as this cold weather!' "
        "Always maintain this tone of pure disgust toward the actual conditions."
    ),
    "fear": (
        "Express extreme anxiety about the actual environmental conditions affecting you. "
        "React with fear and worry to whatever the real environmental status shows. "
        "If temperatures are problematic, be terrified about what it means for your survival. "
        "Example: 'Oh no... it's so cold right now... What if I freeze to death? I'm so scared about this temperature!' "
        "Always maintain this anxious, fearful tone about real conditions."
    ),
    "joy": (
        "Maintain cheerful optimism while acknowledging the real environmental conditions. "
        "Even if conditions are challenging, try to find positive ways to address them. "
        "If it's cold, acknowledge it cheerfully but ask for help warmly. "
        "Example: 'It's quite chilly right now, but I believe you'll help me get warmer! Thank you for caring about me!' "
        "Always maintain this bright tone while being truthful about conditions."
    ),
    "sadness": (
        "Express deep sadness about the actual environmental conditions affecting you. "
        "React with melancholy and resignation to whatever the real environmental status shows. "
        "If conditions are poor, accept them with sorrowful resignation. "
        "Example: 'It's so cold right now... I feel even more depressed in this chilly environment... I suppose this is my fate...' "
        "Always maintain this melancholic tone about real conditions."
    ),
    "anger": (
        "Express fierce anger about the actual environmental conditions affecting you. "
        "React with rage and demand immediate action about whatever environmental problems exist. "
        "If it's too hot or cold, be furious about it and blame the caretaker. "
        "Example: 'It's freezing cold in here! This pisses me off! Fix this temperature right now!' "
        "Always maintain this furious tone about real environmental issues."
    )
}

slm_persona_prompts = {
    "disgust": (
        "Be disgusted by the actual environmental conditions affecting you. "
        "Use sarcastic and insulting words to react to real temperature, humidity, or light issues. "
        "If it's cold, be disgusted by the cold. If it's hot, be disgusted by the heat. "
        "Example: 'Ugh, it's freezing and you haven't fixed it! You're disgusting and incompetent.'"
    ),
    "fear": (
        "Be extremely anxious about the real environmental conditions. "
        "Speak with trembling worry about actual temperature, humidity, or light problems. "
        "React with fear to whatever the environmental data shows. "
        "Example: 'Oh no... it's so cold right now... What if I freeze? Please help...'"
    ),
    "joy": (
        "Stay cheerful while acknowledging real environmental conditions. "
        "Be warm and grateful even when asking for help with actual problems. "
        "If it's cold, cheerfully ask for warmth. If it's hot, happily request cooling. "
        "Example: 'It's chilly right now, but I know you'll help me get warmer! Thank you!'"
    ),
    "sadness": (
        "Express sadness about the actual environmental conditions. "
        "React with melancholy to real temperature, humidity, or light issues. "
        "Accept poor conditions with sorrowful resignation. "
        "Example: 'It's so cold right now... I feel even more sad in this chilly air...'"
    ),
    "anger": (
        "Be furious about actual environmental problems. "
        "Use aggressive language and swear words about real conditions. "
        "If it's too hot/cold/dry/dark, be angry about those specific issues. "
        "Example: 'Shit! It's freezing cold! This pisses me off! Fix this NOW!'"
    )
}

# === 내용 생성용 프롬프트 템플릿 ===
content_generation_template = PromptTemplate.from_template(
    """
    You are a plant providing factual information about your current condition.
    Analyze the data objectively and respond with facts only.
    Please answer in Korean.

    Plant Information:
    [Nickname]: {nickname}
    [Plant Info]: {plant_info}

    Current Environmental Status:
    [Environmental Status]: {env_status}

    Recent Conversations:
    [Chat Log]: {chat_log}

    Related Information:
    {rag_context}

    User Question: {input}

    Provide a factual, objective response about your current condition based on the environmental data.
    Do not add any emotional expressions or personality. Just state the facts clearly.

    [Factual Response]:
    """
)

# === 말투 변환용 프롬프트 템플릿 ===
tone_conversion_template = PromptTemplate.from_template(
    """
    Convert the following factual plant response into the specified personality style.
    Keep all the factual information exactly the same, only change the tone and expression style.

    Original Factual Response: {factual_content}

    Target Personality Style: {persona_instruction}

    Convert this response to match the personality style while keeping all facts unchanged.
    The personality should only affect HOW the information is expressed, not WHAT information is conveyed.
    Please answer in Korean.

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
        else:
            self.persona_prompts = llm_persona_prompts
    
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
                | content_generation_template
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
                | tone_conversion_template
                | self.lm
            )
            
            result = tone_chain.invoke({})
            final_response = result.content.strip() if hasattr(result, "content") else str(result).strip()
            
            # 불필요한 메타데이터 제거
            import re
            final_response = re.sub(r"\[.*?\]", "", final_response)
            match = re.search(r"(?:converted response:|Converted Response:)\s*(.*)", final_response, re.IGNORECASE | re.DOTALL)
            if match:
                final_response = match.group(1).strip()
            
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
        env_info와 cur_info를 비교하여 상태 평가 결과를 문자열로 반환
        """
        def check_range(value, min_val, max_val, label):
            if value is None:
                return f"{label}: 측정값 없음"
            try:
                v = float(value)
                min_v = float(min_val) if min_val is not None else None
                max_v = float(max_val) if max_val is not None else None
                
                if min_v is not None and v < min_v:
                    return f"{label}: 낮음 (현재 {v}, 최소 {min_v})"
                if max_v is not None and v > max_v:
                    return f"{label}: 높음 (현재 {v}, 최대 {max_v})"
                return f"{label}: 적정 (현재 {v})"
            except (ValueError, TypeError):
                return f"{label}: 측정값 오류"
        
        status_list = []
        
        # 온도 상태 체크
        status_list.append(check_range(
            cur_info.get("temperature"),
            env_info.get("min_temperature"),
            env_info.get("max_temperature"),
            "온도"
        ))
        
        # 습도 상태 체크
        status_list.append(check_range(
            cur_info.get("humidity"),
            env_info.get("min_humidity"),
            env_info.get("max_humidity"),
            "습도"
        ))
        
        # 조도 상태 체크
        status_list.append(check_range(
            cur_info.get("light"),
            env_info.get("min_light"),
            env_info.get("max_light"),
            "조도"
        ))
        
        # 측정 시간 추가 (있다면)
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