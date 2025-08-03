import os
import re
import json
from dotenv import load_dotenv
from typing import TypedDict, Literal, Optional, Dict, Any

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
    max_tokens=512,
)

# ======================== 상태 정의 ========================
class PlantQAState(TypedDict):
    input: str
    database_result: Optional[str]
    rag_result: Optional[str]
    web_result: Optional[str]
    final_response: Optional[str]

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
            """
            Answer in Korean based on the provided question, SQL query, and SQL result.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}

            Cuation:
            - Use natural Korean language
            - Include plant names and key features if available
            - Provide specific and helpful information
            - Do not mention sources or documents, answer in a conversational manner
            
            Answer:
            """
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
    
    def search_database(self, question: str) -> Optional[str]:
        """데이터베이스 검색 실행"""
        if not self.check_database_content(question):
            return None
        
        try:
            chain = (
                RunnablePassthrough.assign(query=self.write_query_chain)
                .assign(
                    query=lambda x: self.extract_sql(x["query"]),
                    result=lambda x: self.execute_query_tool.invoke({"query": x["query"]})
                )
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
            "You are an expert in plant care and management. "
            "Answer accurately and naturally in Korean based on the provided context. "
            "If you don't know, say you don't know. "
            "Provide clear and helpful answers to plant-related questions. "
            "Do not mention sources or documents, answer in a conversational manner."
            "\n\nContext: {context}"            
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
    except Exception as e:
        print(f"RAG 시스템 초기화 오류: {e}")
        return None

# ======================== 웹 검색 ========================
def web_search(query: str, max_results: int = 3) -> str:
    """DuckDuckGo를 이용한 웹 검색"""
    try:
        ddgs = DDGS()
        search_query = f"{query} 식물 관리 키우기"
        results = ddgs.text(search_query, max_results=max_results)
        
        if not results:
            return None
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No content')
            formatted_results.append(f"{i}. {title}\n{body[:300]}...")
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        print(f"웹 검색 중 오류가 발생했습니다: {e}")
        return None

# ======================== 멀티에이전트 노드 ========================
# 전역 변수로 핸들러들 초기화
db_handler = None
rag_chain = None

def initialize_handlers():
    """핸들러 초기화"""
    global db_handler, rag_chain
    
    # 데이터베이스 핸들러 초기화
    if os.path.exists("./data/leaf.db"):
        try:
            db_handler = DatabaseQueryHandler("./data/leaf.db", os.getenv("GROQ_API_KEY"))
            # print("데이터베이스 핸들러 초기화 완료")
        except Exception as e:
            print(f"데이터베이스 핸들러 초기화 오류: {e}")
            db_handler = None
    else:
        print("데이터베이스 파일을 찾을 수 없습니다.")
    
    # RAG 시스템 초기화
    rag_chain = initialize_rag()
    if rag_chain:
        print("RAG 시스템 초기화 완료")
    else:
        print("RAG 시스템 초기화 실패")

def database_search_node(state: PlantQAState) -> PlantQAState:
    """데이터베이스 검색 노드"""
    if db_handler:
        try:
            result = db_handler.search_database(state["input"])
            state["database_result"] = result
            # if result:
            #     print("데이터베이스에서 결과를 찾았습니다.")
        except Exception as e:
            print(f"데이터베이스 검색 오류: {e}")
            state["database_result"] = None
    else:
        state["database_result"] = None
    return state

def rag_search_node(state: PlantQAState) -> PlantQAState:
    """RAG 검색 노드"""
    # 데이터베이스에서 결과를 찾았으면 RAG 검색 건너뛰기
    if state["database_result"]:
        state["rag_result"] = None
        return state
    
    if rag_chain:
        try:
            result = rag_chain.invoke({"input": state["input"]})
            answer = result.get("answer", "")
            
            # 답변이 유의미한지 확인
            if answer and not any(keyword in answer.lower() for keyword in ["모르겠", "없습니다", "찾을 수 없", "정보가 없"]):
                state["rag_result"] = answer
                print("RAG 시스템에서 결과를 찾았습니다.")
            else:
                state["rag_result"] = None
        except Exception as e:
            print(f"RAG 검색 오류: {e}")
            state["rag_result"] = None
    else:
        state["rag_result"] = None
    
    return state

def web_search_node(state: PlantQAState) -> PlantQAState:
    """웹 검색 노드"""
    # 이미 다른 소스에서 결과를 찾았으면 웹 검색 건너뛰기
    if state["database_result"] or state["rag_result"]:
        state["web_result"] = None
        return state
    
    try:
        result = web_search(state["input"])
        state["web_result"] = result
        # if result:
        #     print("웹 검색에서 결과를 찾았습니다.")
    except Exception as e:
        print(f"웹 검색 오류: {e}")
        state["web_result"] = None
    
    return state

def response_generator_node(state: PlantQAState) -> PlantQAState:
    """응답 생성 노드"""
    # 가장 우선순위가 높은 정보 소스 선택
    knowledge_source = ""
    source_type = ""
    
    if state["database_result"]:
        knowledge_source = state["database_result"]
        source_type = "데이터베이스"
    elif state["rag_result"]:
        knowledge_source = state["rag_result"]
        source_type = "문서"
    elif state["web_result"]:
        knowledge_source = state["web_result"]
        source_type = "웹 검색"
    
    if knowledge_source:
        # 검색된 정보가 있는 경우
        prompt = f"""
        User asked a question about plants.
        Answer naturally and helpfully in Korean based on the following information.

        User question: {state['input']}

        Reference information ({source_type}):
        {knowledge_source}

        Answering guidelines:
        - Answer in natural Korean as if having a conversation
        - Provide specific and practical information
        - Do not mention sources or documents, answer naturally
        - Clearly explain plant care or characteristics
        - Include additional tips or precautions if necessary

        Answer:
        """
    else:
        # 검색된 정보가 없는 경우
        prompt = f"""
        User asked a question about plants.
        Answer naturally and helpfully in Korean based on general plant care knowledge.

        User question: {state['input']}

        There is no specific information available from the database, documents, or web search.
        Please provide a helpful answer based on general plant care knowledge.
        If you cannot find specific information, explain politely and suggest alternatives.

        Answer:
        """
    
    try:
        response = lm.invoke(prompt)
        state["final_response"] = response.content.strip()
    except Exception as e:
        state["final_response"] = f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {e}"
    
    return state

# ======================== 그래프 구성 ========================
def create_multiagent_graph():
    """멀티에이전트 그래프 생성"""
    graph = StateGraph(PlantQAState)

    # 노드 추가
    graph.add_node("database_search", database_search_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("response_generator", response_generator_node)

    # 엣지 연결 (순차적으로 실행)
    graph.add_edge("database_search", "rag_search")
    graph.add_edge("rag_search", "web_search")
    graph.add_edge("web_search", "response_generator")

    # 시작점과 끝점 설정
    graph.set_entry_point("database_search")
    graph.set_finish_point("response_generator")

    return graph.compile()

# ======================== 메인 실행 함수 ========================
def run_plant_qa_chatbot(user_input: str) -> dict:
    """식물 Q&A 챗봇 실행"""
    
    # 핸들러 초기화
    initialize_handlers()
    
    # 그래프 생성
    app = create_multiagent_graph()
    
    # 입력 전처리
    cleaned_input = re.sub(r"[^\w\sㄱ-힣?!.,]", "", user_input.strip())
    
    # print(f"질문: {cleaned_input}")
    
    # 멀티에이전트 실행
    try:
        output = app.invoke({
            "input": cleaned_input,
            "database_result": None,
            "rag_result": None,
            "web_result": None,
            "final_response": None
        })
        
        return output
    except Exception as e:
        return {"final_response": f"죄송합니다. 처리 중 오류가 발생했습니다: {e}"}

# ======================== 실행 예시 ========================
if __name__ == "__main__":
    import time
    
    # 테스트 질문들
    test_questions = [
        "여름에 기르기 좋은 실내 식물은 뭐가 있을까?",
        # "몬스테라 키우는 방법을 알려줘",
        # "물을 너무 많이 줘서 뿌리가 썩었는데 어떻게 해야할까?",
        # "햇빛이 잘 안 드는 곳에서도 잘 자라는 식물 추천해줘"
    ]
    
    for question in test_questions:
        print("=" * 80)
        start_time = time.time()
        
        result = run_plant_qa_chatbot(question)
        
        print(f"답변: {result.get('final_response', '응답이 없습니다.')}")
        print(f"소요 시간: {time.time() - start_time:.2f}초")
        print()