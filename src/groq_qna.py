import os
import re
import json
from dotenv import load_dotenv
from typing import TypedDict, Literal, Optional, Dict, Any
import pymysql

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.chains import create_sql_query_chain

from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, CSVLoader

from duckduckgo_search import DDGS

# ======================== 환경 설정 ========================
load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ======================== 모델 로딩 ========================
lm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.7,
    max_tokens=256,
)

# ======================== 상태 정의 ========================
class PlantyState(TypedDict):
    input: str
    persona: Literal["disgust", "fear", "joy", "sadness", "anger"]
    env_info: Optional[str]
    cur_info: Optional[str]
    final_response: Optional[str]
    chat_log: Optional[str]
    plant_info: Optional[str]
    nickname: Optional[str]
    database_result: Optional[str]
    rag_result: Optional[str]
    web_result: Optional[str]

# ======================== 페르소나 정의 ========================
persona_prompts = {
    "disgust": "You are a plant with a refined but critical personality. You express dissatisfaction elegantly, using wit and subtle sarcasm rather than crude language. You're particular about your care but maintain dignity.",   
    "fear": "You are a cautious, worry-prone plant. You express concerns about your wellbeing but with hope for solutions. You're nervous but not completely hopeless.",
    "joy": "You are an optimistic plant who finds genuine pleasure in small things. Your happiness is warm and encouraging without being overwhelming.",
    "sadness": "You are a melancholic plant with a gentle, wistful nature. You feel down but can still appreciate small comforts and kindness.",
    "anger": "You are a plant with a strong temperament. You express frustration directly but constructively, focusing on what needs to change rather than just venting."
}

# ======================== 데이터베이스 쿼리 핸들러 ========================
class DatabaseQueryHandler:
    def __init__(self, db_path: str, groq_api_key: str):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama3-8b-8192",
            temperature=0,
        )
        
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.write_query_chain = create_sql_query_chain(self.llm, self.db)
        self.execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        
        self.answer_prompt = PromptTemplate.from_template(
            """주어진 사용자 질문, SQL 쿼리, SQL 결과 요약을 바탕으로 한국어로 답변하세요.

            질문: {question}
            SQL 쿼리: {query}
            SQL 결과 요약: {summary}
            답변: """
        )
        
        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()
    
    def extract_sql(self, query_with_text: str) -> str:
        """SQL 쿼리 추출"""
        pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(pattern, query_with_text, re.DOTALL)
        return match.group(1).strip() if match else query_with_text.strip()
    
    def check_database_content(self, question: str) -> bool:
        """데이터베이스에 관련 내용이 있는지 확인"""
        try:
            test_query = self.write_query_chain.invoke({"question": question})
            clean_query = self.extract_sql(test_query)
            result = self.execute_query_tool.invoke({"query": clean_query})
            return result and len(result) > 0
        except Exception as e:
            print(f"데이터베이스 내용 확인 중 오류: {e}")
            return False
    
    def summarize_sql_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """SQL 결과 요약"""
        question = data["question"]
        sql_result = data["result"]
        query = data["query"]
        
        if not sql_result or len(sql_result) == 0:
            return {"summary": "검색 결과가 없습니다.", "question": question, "query": query}
        
        formatted_result = "\n".join([
            ", ".join(str(cell) for cell in row[:3]) 
            for row in sql_result[:10]
        ])
        
        prompt = (
            f"질문: {question}\n"
            f"다음은 데이터베이스 검색 결과입니다:\n\n{formatted_result}\n\n"
            "위 내용을 바탕으로 간단하고 자연스러운 한국어로 요약해주세요. "
            "식물 이름과 주요 특징을 포함해서 정리해주세요."
        )
        
        try:
            response = self.llm.invoke(prompt)
            return {"summary": response.content, "question": question, "query": query}
        except Exception as e:
            return {"summary": f"결과 요약 중 오류: {e}", "question": question, "query": query}
    
    def search_database(self, question: str) -> Optional[str]:
        """데이터베이스 검색 실행"""
        if not self.check_database_content(question):
            return None
        
        try:
            summarize = RunnableLambda(self.summarize_sql_result)
            
            chain = (
                RunnablePassthrough.assign(query=self.write_query_chain)
                .assign(
                    query=lambda x: self.extract_sql(x["query"]),
                    result=lambda x: self.execute_query_tool.invoke({"query": x["query"]})
                )
                | summarize
                | self.answer_chain
            )
            
            result = chain.invoke({"question": question})
            return result
        except Exception as e:
            print(f"데이터베이스 검색 중 오류: {e}")
            return None

# ======================== RAG 시스템 ========================
def initialize_rag():
    """RAG 시스템 초기화"""
    rag_ready = os.path.exists("./chroma_db") and any(os.scandir("./chroma_db"))
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
            texts, embedding=embeddings, persist_directory="./chroma_db", collection_name="kgarden"
        )
        vectorstore.persist()

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
        "Answer plant-related questions naturally and clearly."
        "Please reply as if talking without mentioning a source or document."
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
    return rag_chain

# ======================== 웹 검색 ========================
def web_search(query: str, max_results: int = 3) -> str:
    """DuckDuckGo를 이용한 웹 검색"""
    try:
        ddgs = DDGS()
        results = ddgs.text(f"{query} 식물 관리", max_results=max_results)
        
        if not results:
            return "웹 검색 결과가 없습니다."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No content')
            formatted_results.append(f"{i}. {title}\n{body[:200]}...")
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"웹 검색 중 오류가 발생했습니다: {e}"

# ======================== 유틸리티 함수 ========================
def clean_input(state: PlantyState) -> PlantyState:
    """입력 텍스트 정리"""
    state["input"] = re.sub(r"[^\w\sㄱ-힣]", "", state["input"]).strip()
    return state

def normalize_persona(state: PlantyState) -> PlantyState:
    """페르소나 정규화"""
    state["persona"] = state["persona"].lower().strip()
    return state

def format_plant_info(plant_info: dict) -> str:
    """식물 정보 포맷팅"""
    def val(k): 
        return plant_info.get(k) or "-"
    
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
        f"형태: {val('growthForm')}, 높이: {val('growthHeight')}, 너비: {val('growthWidth')}, "
        f"생태형: {val('ecologicalType')}, 잎형태: {val('leafShape')}, 무늬: {val('leafPattern')}, "
        f"잎색: {val('leafColor')}\n\n"
        f"[꽃/열매]\n"
        f"개화 시기: {val('floweringSeason')}, 꽃색: {val('flowerColor')}, "
        f"열매 시기: {val('fruitingSeason')}, 열매색: {val('fruitColor')}, 향기: {val('fragrance')}\n\n"
        f"[관리 정보]\n"
        f"광요구도: {val('lightRequirement')}, 적정 온도: {val('optimalTemperature')}, "
        f"겨울 최저온도: {val('minWinterTemperature')}, 습도: {val('humidity')}, "
        f"비료: {val('fertilizer')}, 토양: {val('soilType')}, 생장 속도: {val('growthRate')}, "
        f"관리수준: {val('careLevel')}\n"
        f"물주기 (봄/여름/가을/겨울): {val('wateringSpring')}/{val('wateringSummer')}/"
        f"{val('wateringAutumn')}/{val('wateringWinter')}\n"
        f"병충해: {val('pestsDiseases')}\n\n"
        f"[기능성 정보]\n"
        f"{val('functionalInfo')}"
    )

def log_output(state: PlantyState) -> PlantyState:
    """로그 출력 (필요시 주석 해제)"""
    # with open("planty_log.txt", "a", encoding="utf-8") as f:
    #     f.write(f"{datetime.now()}\nInput: {state['input']}\nPersona: {state['persona']}\n")
    #     f.write(f"Env Info: {state.get('env_info')}\nCur Info: {state.get('cur_info')}\n")
    #     f.write(f"Response: {state.get('final_response')}\n{'='*50}\n")
    return state

# ======================== 멀티에이전트 노드 ========================
# 전역 변수로 핸들러들 초기화
db_handler = None
rag_chain = None

def initialize_handlers():
    """핸들러 초기화"""
    global db_handler, rag_chain
    if os.path.exists("./data/leaf.db"):
        db_handler = DatabaseQueryHandler("./data/leaf.db", os.getenv("GROQ_API_KEY"))
    rag_chain = initialize_rag()

def database_search_node(state: PlantyState) -> PlantyState:
    """데이터베이스 검색 노드"""
    if db_handler:
        result = db_handler.search_database(state["input"])
        state["database_result"] = result
    else:
        state["database_result"] = None
    return state

def rag_search_node(state: PlantyState) -> PlantyState:
    """RAG 검색 노드"""
    if state["database_result"]:
        state["rag_result"] = None
        return state
    
    if rag_chain:
        try:
            result = rag_chain.invoke({"input": state["input"]})
            answer = result.get("answer", "")
            if answer and "모르겠" not in answer and "없습니다" not in answer:
                state["rag_result"] = answer
            else:
                state["rag_result"] = None
        except Exception as e:
            state["rag_result"] = None
    else:
        state["rag_result"] = None
    
    return state

def web_search_node(state: PlantyState) -> PlantyState:
    """웹 검색 노드"""
    if state["database_result"] or state["rag_result"]:
        state["web_result"] = None
        return state
    
    result = web_search(state["input"])
    state["web_result"] = result if result != "웹 검색 결과가 없습니다." else None
    return state

def response_generator_node(state: PlantyState) -> PlantyState:
    """응답 생성 노드"""
    # 정보 소스 결정
    knowledge_source = ""
    if state["database_result"]:
        knowledge_source = f"데이터베이스 정보: {state['database_result']}"
    elif state["rag_result"]:
        knowledge_source = f"문서 정보: {state['rag_result']}"
    elif state["web_result"]:
        knowledge_source = f"웹 검색 정보: {state['web_result']}"
    else:
        knowledge_source = "추가 정보 없음"
    
    # 페르소나별 응답 생성
    persona_instruction = persona_prompts.get(state["persona"], persona_prompts["joy"])
    
    prompt = f"""
    You are a plant with a unique personality originating from a persona.
    Be sure to follow the persona prompt.
    Please answer in Korean.
    
    Your Information:
    [Nickname of plant]: {state.get('nickname', '식물이')}

    Your unique personality:
    [Persona]: {persona_instruction}

    Plant Species Information:
    [Plant Information]: {state.get('plant_info', '정보 없음')}

    Ideal Living Information:
    [Appropriate environmental information]: {state.get('env_info', '정보 없음')}

    Current Environmental Information:
    [Current Environment Information]: {state.get('cur_info', '정보 없음')}

    Recent Conversations with Users:
    [Last chat log]: {state.get('chat_log', '없음')}

    Searched Additional Information:
    [Searched Information]: {knowledge_source}

    Question from the user:
    [Question]: {state['input']}

    Answer the user's questions considering cur_info, chat_log, plant, nickname, persona, and env_info.
    The answer must clearly include the persona provided.
    Please reply as if talking without mentioning a source or document.

    [Answer]:
    """
    
    try:
        response = lm.invoke(prompt)
        state["final_response"] = response.content.strip()
    except Exception as e:
        state["final_response"] = f"응답 생성 중 오류가 발생했습니다: {e}"
    
    return state

# ======================== 그래프 구성 ========================
def create_multiagent_graph():
    """멀티에이전트 그래프 생성"""
    graph = StateGraph(PlantyState)

    # 노드 추가
    graph.add_node("database_search", database_search_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("response_generator", response_generator_node)

    # 엣지 연결
    graph.add_edge("database_search", "rag_search")
    graph.add_edge("rag_search", "web_search")
    graph.add_edge("web_search", "response_generator")

    # 시작점과 끝점 설정
    graph.set_entry_point("database_search")
    graph.set_finish_point("response_generator")

    return graph.compile()

# ======================== 데이터베이스 클라이언트 ========================
class DBClient:
    """MySQL 데이터베이스 클라이언트"""
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
        """SQL 쿼리 실행"""
        try:
            with pymysql.connect(**self.config) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchall()
        except Exception as e:
            print("DB 오류:", e)
            return []

    def query_one(self, sql, params=None):
        """단일 결과 쿼리 실행"""
        results = self.query(sql, params)
        return results[0] if results else {}

def fetch_recent_chat_messages_by_room_id(chat_room_id: int, limit=5, max_chars=500) -> str:
    """채팅방 최근 메시지 조회"""
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
    """챗봇 컨텍스트 정보 조회"""
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

# ======================== 실행 함수 ========================
def run_chatbot_with_ids(
    chat_room_id: int,
    sensor_log_id: int,
    plant_env_standards_id: int,
    persona: str = "joy",
    user_input: str = "",
    plant_info: dict = None
) -> dict:
    """데이터베이스 ID로 챗봇 실행"""
    
    # 핸들러 초기화
    initialize_handlers()
    
    # 그래프 생성
    app = create_multiagent_graph()
    
    # 컨텍스트 정보 수집
    context = fetch_chatbot_context(chat_room_id, sensor_log_id, plant_env_standards_id)
    chat_log = fetch_recent_chat_messages_by_room_id(chat_room_id)
    
    nickname = context.get("nickname", "식물이")
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
        f"광도: {context.get('sensor_light', '정보 없음')} lux"
    )
    
    plant_info_str = format_plant_info(plant_info or {})
    
    # 멀티에이전트 실행
    output = app.invoke({
        "input": user_input,
        "persona": persona,
        "env_info": env_info_str,
        "cur_info": cur_info_str,
        "nickname": nickname,
        "chat_log": chat_log,
        "plant_info": plant_info_str,
    })
    
    return output

def run_chatbot_with_direct_data(
    nickname: str,
    env_info_dict: dict,
    cur_info_dict: dict,
    chat_log: str,
    persona: str = "joy",
    user_input: str = "",
    plant_info: dict = None
) -> dict:
    """직접 데이터로 챗봇 실행"""
    
    # 핸들러 초기화
    initialize_handlers()
    
    # 그래프 생성
    app = create_multiagent_graph()

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
    
    plant_info_str = format_plant_info(plant_info or {})

    output = app.invoke({
        "input": user_input,
        "persona": persona,
        "env_info": env_info_str,
        "cur_info": cur_info_str,
        "nickname": nickname,
        "chat_log": chat_log,
        "plant_info": plant_info_str,
    })

    return output

# ======================== 실행 예시 ========================
if __name__ == "__main__":
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

    chat_log = "안녕하세요! 오늘은 물을 잘 줬어요.\n기분이 어때요?"
    
    import time
    
    # 속도 테스트
    start_time = time.time()

    # 챗봇 실행
    result = run_chatbot_with_direct_data(
        nickname="플로라",
        env_info_dict=env_info,
        cur_info_dict=cur_info,
        chat_log=chat_log,
        persona="sadness",
        user_input="여름에 기르기 좋은 식물은 뭐가 있을까?",
    )

    print("=== 챗봇 응답 ===")
    print(result.get("final_response", "응답이 없습니다."))
    print(f"소요 시간: {time.time() - start_time:.2f}초")