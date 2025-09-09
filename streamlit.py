from dotenv import load_dotenv
import os

import streamlit as st
import time
import datetime

from src.groq_qa import run_llm_plant_qa_chatbot
from src.groq_chatbot import run_llm_chatbot_with_direct_data

from src.slm_qa import run_slm_plant_qa_chatbot
from src.slm_chatbot import run_slm_chatbot_with_direct_data

# ======================== 환경 설정 ========================

load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

# ======================== 챗봇 설정 ========================

st.set_page_config(page_title="Planty Agent", page_icon="🌱")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_choice" not in st.session_state:
    st.session_state.api_choice = None

# ======================== 사이드바 ========================

def sidebar():
    with st.sidebar:
        # ========================= 챗봇 설정 ========================= 
        st.subheader("🤖 챗봇 설정")

        # api 선택
        api_choice = st.selectbox(
            "챗봇 선택",
            options=["Chatroom 1", "Chatroom 2", "Chatroom 3", "Chatroom 4"],
            help="사용할 챗봇을 선택하세요.",
        )
        
        if api_choice == "Chatroom 1":
            st.session_state.api = st.secrets['GROQ_API_KEY_NO1']
        elif api_choice == "Chatroom 2":
            st.session_state.api = st.secrets['GROQ_API_KEY_NO2']
        elif api_choice == "Chatroom 3":
            st.session_state.api = st.secrets['GROQ_API_KEY_NO3']
        elif api_choice == "Chatroom 4":
            st.session_state.api = st.secrets['GROQ_API_KEY_NO4']
        else:
            st.error("잘못된 선택입니다.")
            return
        st.session_state.api_choice = api_choice

        # 챗봇 종류 선택 
        col1, col2 = st.columns(2)
        with col1:            
            chatbot_model = st.radio(
                "챗봇 모델",
                options=["SLM", "LLM"],
                help="사용할 챗봇 모델을 선택하세요.",
            )
            st.session_state.chatbot_model = chatbot_model
        with col2:
            chatbot_mode = st.radio(
                "챗봇 모드",
                options=["성격", "질의응답"],
                help="챗봇의 동작 방식을 선택하세요.",
            )
            st.session_state.chatbot_mode = chatbot_mode

        # ========================= 식물 정보 설정 ========================= 
        st.divider()
        st.subheader("🌱 식물 정보 설정")

        # 식물 종류
        plant_type = st.selectbox(
            "식물 종류",
            options=["몬스테라", "가울테리아", "개운죽", "러브체인", "숙근이베리스", "시서스", 
                     "스킨답서스", "아이비", "히포에스테스", "호야"],
            help="해당 식물의 종류를 선택하세요.",
        )
        st.session_state.plant_type = plant_type

        col3, col4 = st.columns(2)
        with col3:
            # 식물 이름
            plant_name = st.text_input("식물 이름", value="테리")
            st.session_state.plant_name = plant_name

            # 식물 환경
            plant_env = st.selectbox(
                "식물 환경",
                options=["적절", "건조", "습함", "추움", "더움"],
            )
            st.session_state.plant_env = plant_env

        with col4:
            # 식물 나이
            plant_age = st.number_input(
                "식물 나이",
                min_value=0,
                value=1,
                step=1,
            )
            st.session_state.plant_age = plant_age

            # 식물 성격
            plant_personality = st.selectbox(
                "식물 성격",
                options=["기쁨이", "슬픔이", "까칠이", "버럭이", "소심이"],
                help="식물의 성격을 선택하세요.",
            )
            st.session_state.plant_personality = plant_personality

        # 대화 기록 삭제 버튼
        if st.button("대화 기록 삭제", type="secondary"):
            st.session_state.messages = []
            st.rerun()

# ======================== 챗봇 ========================

def run_persona(type, user_input: str) -> str:
    env_info = {
        "max_humidity": 80,
        "max_light": 15000,
        "max_temperature": 30,
        "min_humidity": 40,
        "min_light": 5000,
        "min_temperature": 15
    }

    # 기본값 = "적절"
    env_type = st.session_state.plant_env
    temperature = (env_info["min_temperature"] + env_info["max_temperature"]) // 2
    humidity = (env_info["min_humidity"] + env_info["max_humidity"]) // 2
    light = (env_info["min_light"] + env_info["max_light"]) // 2

    if env_type == "건조":
        humidity = env_info["min_humidity"] - 5
    elif env_type == "습함":
        humidity = env_info["max_humidity"] + 5
    elif env_type == "추움":
        temperature = env_info["min_temperature"] - 2
    elif env_type == "더움":
        temperature = env_info["max_temperature"] + 2
    # "적절"이면 그대로 사용

    cur_info = {
        "temperature": temperature,
        "humidity": humidity,
        "light": light,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    chat_log = "\n".join([m["content"] for m in st.session_state.messages if m["role"] == "user"])

    personality_map = {
        "기쁨이": "joy",
        "슬픔이": "sadness",
        "까칠이": "disgust",
        "버럭이": "anger",
        "소심이": "fear"
    }

    personality = personality_map.get(st.session_state.plant_personality, "joy")

    if type == "SLM":
        result = run_slm_chatbot_with_direct_data(
            nickname=st.session_state.plant_name,
            env_info_dict=env_info,
            cur_info_dict=cur_info,
            chat_log=chat_log,
            persona=personality,
            user_input=user_input
        )
    else:
        result = run_llm_chatbot_with_direct_data(
            nickname=st.session_state.plant_name,
            env_info_dict=env_info,
            cur_info_dict=cur_info,
            chat_log=chat_log,
            persona=personality,
            user_input=user_input
        )
    return result

def get_chatbot_response(user_input: str) -> str:
    if st.session_state.chatbot_mode == "성격":
        result = run_persona(st.session_state.chatbot_model, user_input)
    else:
        if st.session_state.chatbot_model == "SLM":
            result = run_slm_plant_qa_chatbot(user_input)
        else:
            result = run_llm_plant_qa_chatbot(user_input) 

    return result.get("final_response", "응답이 없습니다.")

def chatbot():
    chat_container = st.container()

    with chat_container:
        # 기존 메시지들 표시
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])


    # 사용자 입력
    user_input = st.chat_input("메시지를 입력하세요...")


    # 사용자가 메시지를 입력했을 때
    if user_input:
        # 사용자 메시지를 세션에 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 사용자 메시지 표시
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
        
        # 봇 응답 생성
        with st.spinner("생각 중..."):
            time.sleep(1)  # 실제 응답을 기다리는 것처럼 보이게 하는 딜레이
            bot_response = get_chatbot_response(user_input)
        
        # 봇 응답을 세션에 추가
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # 봇 응답 표시
        with chat_container:
            with st.chat_message("assistant"):
                st.write(bot_response)
        
        # 페이지 새로고침으로 최신 메시지가 보이도록
        st.rerun()

# ======================== 챗봇 실행 ========================

if __name__ == "__main__":
    st.title("🌱 Planty Agent")
    sidebar()
    chatbot()

