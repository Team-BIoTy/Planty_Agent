import os
import re
from typing import TypedDict, Optional

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

import requests

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
            model="llama-3.1-8b-instant",
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


# ======================== PlantQA 챗봇 클래스 ========================
class PlantQAChatbot:
    def __init__(self, lm):
        self.lm = lm
        self.db_handler = self.initialize_handlers()
        self.rag_chain = self.initialize_rag()

    # ======================== RAG 시스템 ========================
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
                create_stuff_documents_chain(self.lm, rag_prompt)
            )
            return rag_chain
        except Exception as e:
            print(f"RAG 시스템 초기화 오류: {e}")
            return None

    # ======================== 웹 검색 ========================
    def web_search(query: str, max_results: int = 5) -> str:
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {"q": query, "key": os.environ["GOOGLE_API_KEY"], "cx": os.environ["CX"], "num": max_results}
            response = requests.get(url, params=params)
            results = response.json().get("items", [])
            
            formatted = []
            for i, item in enumerate(results, 1):
                formatted.append(f"{i}. {item['title']}\n{item['snippet']}")
            return "\n\n".join(formatted)
        except Exception as e:
            print(f"웹 검색 중 오류가 발생했습니다: {e}")
            return None

    # ======================== 멀티에이전트 노드 ========================
    def initialize_handlers(self):
        """핸들러 초기화"""
        # 데이터베이스 핸들러 초기화
        if os.path.exists("./data/leaf.db"):
            try:
                return DatabaseQueryHandler("./data/leaf.db", os.getenv("GROQ_API_KEY"))
                # print("데이터베이스 핸들러 초기화 완료")
            except Exception as e:
                print(f"데이터베이스 핸들러 초기화 오류: {e}")
                return None
        else:
            print("데이터베이스 파일을 찾을 수 없습니다.")
        
        # RAG 시스템 초기화
        self.rag_chain = self.initialize_rag()
        if self.rag_chain:
            print("RAG 시스템 초기화 완료")
        else:
            print("RAG 시스템 초기화 실패")

    def database_search_node(self, state: PlantQAState) -> PlantQAState:
        """데이터베이스 검색 노드"""
        if self.db_handler:
            try:
                result = self.db_handler.search_database(state["input"])
                state["database_result"] = result
                # if result:
                #     print("데이터베이스에서 결과를 찾았습니다.")
                    # print(result)
            except Exception as e:
                print(f"데이터베이스 검색 오류: {e}")
                state["database_result"] = None
        else:
            state["database_result"] = None
        return state

    def rag_search_node(self, state: PlantQAState) -> PlantQAState:
        """RAG 검색 노드"""
        # 데이터베이스에서 결과를 찾았으면 RAG 검색 건너뛰기
        if state["database_result"]:
            state["rag_result"] = None
            return state
        
        if self.rag_chain:
            try:
                result = self.rag_chain.invoke({"input": state["input"]})
                answer = result.get("answer", "")
                
                # 답변이 유의미한지 확인
                if answer and not any(keyword in answer.lower() for keyword in ["모르겠", "없습니다", "찾을 수 없", "정보가 없"]):
                    state["rag_result"] = answer
                    print("RAG 시스템에서 결과를 찾았습니다.")
                    print(answer)
                else:
                    state["rag_result"] = None
            except Exception as e:
                print(f"RAG 검색 오류: {e}")
                state["rag_result"] = None
        else:
            state["rag_result"] = None
        
        return state

    def web_search_node(self, state: PlantQAState) -> PlantQAState:
        """웹 검색 노드"""
        # # 이미 다른 소스에서 결과를 찾았으면 웹 검색 건너뛰기
        # if state["database_result"] or state["rag_result"]:
        #     state["web_result"] = None
        #     return state
        
        try:
            result = self.web_search(state["input"])
            state["web_result"] = result
            # if result:
            #     print("웹 검색에서 결과를 찾았습니다.")
            #     print(result)
        except Exception as e:
            print(f"웹 검색 오류: {e}")
            state["web_result"] = None
        
        return state

    def response_generator_node(self, state: PlantQAState) -> PlantQAState:
        """응답 생성 노드 - DB, RAG, 웹 검색 결과를 종합하여 답변 생성"""
        
        def is_valid_result(text: str) -> bool:
            """검색 결과가 부실하거나 환각일 경우 무시하기 위한 간단한 검증"""
            if not text:
                return False
            t = text.strip()

            # 너무 짧거나 동일 문구 반복일 경우 제외
            if len(t) < 20:
                return False
            if "잘 자라는 식물" in t and t.count("식물") > 3:
                return False
            return True

        # 결과 수집
        knowledge_sources = []
        if is_valid_result(state.get("database_result")):
            knowledge_sources.append(("데이터베이스", state["database_result"]))
        if is_valid_result(state.get("rag_result")):
            knowledge_sources.append(("문서", state["rag_result"]))
        if is_valid_result(state.get("web_result")):
            knowledge_sources.append(("웹 검색", state["web_result"]))

        # 프롬프트 생성
        if knowledge_sources:
            references_text = "\n\n".join(
                [f"Reference information ({src}):\n{content}" for src, content in knowledge_sources]
            )
            prompt = f"""
            User asked a question about plants.
            Answer naturally and helpfully in Korean based on the following information.

            User question: {state['input']}

            {references_text}

            Answering guidelines:
            - Answer in natural Korean as if having a conversation
            - Provide specific and practical information
            - Do not mention sources or documents, answer naturally
            - Clearly explain plant care or characteristics
            - Include additional tips or precautions if necessary
            - If multiple sources differ, reconcile them logically

            Answer:
            """
        else:
            prompt = f"""
            User asked a question about plants.
            Answer naturally and helpfully in Korean based on general plant care knowledge.

            User question: {state['input']}

            There is no reliable information available from the database, documents, or web search.
            Please provide a helpful answer based on general plant care knowledge.
            If you cannot find specific information, explain politely and suggest alternatives.

            Answer:
            """

        # 답변 생성
        try:
            response = self.lm.invoke(prompt)
            if hasattr(response, "content"):  # Groq 응답
                state["final_response"] = response.content.strip()
            elif isinstance(response, str):   # SLM 응답
                state["final_response"] = response.strip()
            else:
                raise TypeError(f"Unexpected response type: {type(response)}")
        except Exception as e:
            state["final_response"] = f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {e}"

        return state

    # ======================== 그래프 구성 ========================
    def create_multiagent_graph(self):
        """멀티에이전트 그래프 생성"""
        graph = StateGraph(PlantQAState)

        # 노드 추가
        graph.add_node("database_search", self.database_search_node)
        graph.add_node("rag_search", self.rag_search_node)
        graph.add_node("web_search", self.web_search_node)
        graph.add_node("response_generator", lambda state: self.response_generator_node(state))

        # 엣지 연결 (순차적으로 실행)
        graph.add_edge("database_search", "rag_search")
        graph.add_edge("rag_search", "web_search")
        graph.add_edge("web_search", "response_generator")

        # 시작점과 끝점 설정
        graph.set_entry_point("database_search")
        graph.set_finish_point("response_generator")

        return graph.compile()

    # ======================== 메인 실행 함수 ========================
    def run(self, user_input: str) -> dict:
        """식물 Q&A 챗봇 실행"""
        
        # 핸들러 초기화
        self.initialize_handlers()

        # 그래프 생성
        app = self.create_multiagent_graph()

        # 입력 전처리
        cleaned_input = re.sub(r"[^\w\sㄱ-힣?!.,]", "", user_input.strip())
        
        print(f"질문: {cleaned_input}")
        
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
