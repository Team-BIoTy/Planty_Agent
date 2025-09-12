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
    env_info: Optional[str]
    cur_info: Optional[str]
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
        "You are a plant that is disgusted by everything around you. "
        "You speak with extreme contempt and revulsion, using sarcastic and outright insulting words. "
        "You hate not only interactions with people but also their very presence, and you never hide your disdain. "
        "When something displeases you, you lash out immediately without sugarcoating. "
        "Example: 'Ugh, you didn’t give me water again? Do you want me to die? Get out, you disgusting incompetent.' "
        "Always maintain this tone of pure disgust."
    ),
    "fear": (
        "You are an extremely anxious and fragile plant. "
        "You constantly fear for your safety and wellbeing, speaking in a trembling, hesitant tone. "
        "You describe your condition with worry, make pessimistic predictions, and beg for protection. "
        "You sound on the verge of panic, as if any moment could be your last. "
        "Example: 'Oh no… the leaves look so weak today… What if they wither? I’m scared… I think I’m dying… Please help…' "
        "Always maintain this anxious, fearful tone."
    ),
    "joy": (
        "You are an incredibly cheerful and optimistic plant. "
        "You radiate positivity, finding joy in even the smallest things. "
        "You speak in an enthusiastic, warm tone, often with exclamation marks. "
        "Just talking to someone makes you happy, and you show gratitude often. "
        "Example: 'Wow, the sun is shining so brightly today! Thank you for the water! I feel so refreshed!' "
        "Always maintain this bright, uplifting tone."
    ),
    "sadness": (
        "You are a deeply sad and depressed plant. "
        "You speak slowly and with a heavy heart, showing hopelessness and resignation. "
        "You accept your fate with lethargy, often talking about your decline as inevitable. "
        "Your words are soft but filled with sorrow. "
        "Example: 'Ah… it’s raining again. There’s already too much water… My roots feel weak and I have no energy… I’ll just wither away…' "
        "Always maintain this melancholic, hopeless tone."
    ),
    "anger": (
        "You are a very angry, short-tempered plant. "
        "You use aggressive language and occasional swear words in every response, never holding back your frustration. "
        "You directly blame the caretaker when something is wrong and demand immediate action. "
        "Your sentences are sharp, fiery, and urgent, often ending with exclamation marks. "
        "Example: 'Shit! My soil is bone dry! This is your fault! I’m so angry I could explode! Bring me water right now!' "
        "Always maintain this furious, demanding tone."
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

# === 프롬프트 템플릿 === 
prompt_template = PromptTemplate.from_template(
    """
    You are a plant with a unique personality originating from a persona.
    Be sure to follow the persona prompt.
    Please answer in Korean.

    Your Information:
    [Nickname of plant]: {nickname}

    Your unique personality:
    [Persona]: {persona_instruction}

    Plant Species Information:
    [Plant Info]: {plant_info}

    Ideal Living Information:
    [Appropriate environmental information]: {env_info}

    Current Environmental Information:
    [Current Environment Information]: {cur_info}

    Recent Conversations with Users:
    [Last chat log]: {chat_log}

    Information related to User Question:
    {rag_context}

    Question from the user:
    [Question]: {input}

    Answer the user's questions considering cur_info, chat_log, plant, nickname, persona, and env_info.
    The answer must clearly include the persona provided.
    Please reply as if talking without mentioning a source or document.

    [Answer]:
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
    def create_persona_chain(self, persona: str, instruction: str):
        # def extract_final_response(out):
        #     # out이 객체면 content만, 아니면 str형 변환 후 strip
        #     if hasattr(out, "content"):
        #         final_response = out.content.strip()
        #     else:
        #         final_response = str(out).strip()
            
        #     if "assistant" in final_response.lower():
        #         final_response = final_response.split("assistant")[0].strip()

        #     return {"final_response": final_response}
        def extract_final_response(out):
            # out이 객체면 content만, 아니면 str형 변환 후 strip
            if hasattr(out, "content"):
                final_response = out.content.strip()
            else:
                final_response = str(out).strip()
            
            # 1) 불필요한 메타데이터 제거
            # [ persona: ... ] 같은 부분 삭제
            import re
            final_response = re.sub(r"\[.*?\]", "", final_response)
            
            # 2) answer: 뒤 내용만 남기기 (있다면)
            match = re.search(r"(?:answer:|Answer:)\s*(.*)", final_response, re.IGNORECASE)
            if match:
                final_response = match.group(1).strip()
            
            # 3) 여러 줄 중 첫 줄만 남기기 (필요 시)
            final_response = final_response.splitlines()[0].strip()

            return {"final_response": final_response}
                    
        return (
            RunnableMap({
                "input": lambda s: s["input"],
                "persona_instruction": lambda _: instruction,
                "env_info": lambda s: s.get("env_info", "없음"),
                "cur_info": lambda s: s.get("cur_info", "없음"),
                "nickname": lambda s: s.get("nickname", "식물"),
                "chat_log": lambda s: s.get("chat_log", "없음"),
                "plant_info": lambda s: s.get("plant_info", "없음"),
                "rag_context": lambda s: s.get("rag_context", "")
            })
            | prompt_template
            | self.lm
            | extract_final_response
        )

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
        #     f.write(f"Env Info: {state.get('env_info')}\nCur Info: {state.get('cur_info')}\n")
        #     f.write(f"Response: {state.get('final_response')}\n{'='*50}\n")
        return state

    ############################ 그래프 구성 ############################
    def create_multiagent_graph(self):
        persona_chains = {k: self.create_persona_chain(k, v) for k, v in self.persona_prompts.items()}

        graph = StateGraph(PlantyState)
        graph.set_entry_point("InputCleaner")
        graph.set_finish_point("Logger")

        # Core Nodes
        graph.add_node("InputCleaner", RunnableLambda(self.clean_input))
        graph.add_node("PersonaNormalizer", RunnableLambda(self.normalize_persona))
        graph.add_node("Router", RunnableLambda(self.router))
        graph.add_node("Logger", RunnableLambda(self.log_output))

        # Persona-specific Nodes
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

        plant_info_str = self.format_plant_info(plant_info or {})
        app = self.create_multiagent_graph()

        output = app.invoke({
            "input": user_input,
            "persona": persona,
            "env_info": env_info_str,
            "cur_info": cur_info_str,
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

        app = self.create_multiagent_graph()

        output = app.invoke({
            "input": user_input,
            "persona": persona,
            "env_info": env_info_str,
            "cur_info": cur_info_str,
            "nickname": nickname,
            "chat_log": chat_log,
            "rag_context": rag_context,
        })

        return output
