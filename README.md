<img alt="Python" src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" height="20"/><img alt="LangChain" src="https://img.shields.io/badge/LangChain-1C3C3C.svg?style=for-the-badge&logo=LangChain&logoColor=white" height="20"/>
<img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-1C3C3C.svg?style=for-the-badge&logo=LangGraph&logoColor=white" height="20"/>
<img alt="MySQL" src="https://img.shields.io/badge/MySQL-4479A1.svg?style=for-the-badge&logo=MySQL&logoColor=white" height="20"/>
<img alt="GitHub" src="https://img.shields.io/badge/GitHub-181717.svg?style=for-the-badge&logo=GitHub&logoColor=white" height="20"/>
<img alt="Notion" src="https://img.shields.io/badge/Notion-000000.svg?style=for-the-badge&logo=Notion&logoColor=white" height="20"/>

</br>

# 🌱 Planty Agent

**Planty Project**에서 사용하는 LLM 기반 Agent입니다. 

식물 환경 센서 정보와 감정 기반 페르소나를 반영하여 자연스럽고 상황에 맞는 대화를 제공합니다.

### 프로젝트 개요
- **전체 개발 기간**: 
- **Agent 설계**: 2025.05.01 - 2025.05.07
- **기능 구현**: 2025.05.14 - 2025.05.31
- **기본 제공 모델**: [yerim00/HyperCLOVAX-SEED-Text-Instruct-1.5B-planty-ia3](yerim00/HyperCLOVAX-SEED-Text-Instruct-1.5B-planty-ia3)


*기본 제공모델은 naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B 모델을 기반으로 파인튜닝한 모델로 파인튜닝 코드는 [Planty_LLM](naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)에서 확인 가능합니다. 

</br>

## 📦 다운로드 및 설치

### 1. 레포지토리 클론

```bash
git clone https://github.com/Team-BIoTy/Planty_Agent.git
cd Planty_Agent
```

### 2. 모델 다운로드

```python
# model_download.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yerim00/HyperCLOVAX-SEED-Text-Instruct-1.5B-planty-ia3",
    local_dir="./HyperCLOVAX-Local",
    local_dir_use_symlinks=True
)
```

* `model_download.py`를 실행하여 로컬에 CLOVAX 모델을 다운로드합니다.

</br>

## ⚙️ 환경 설정

### 1. 가상환경 설정 (권장)

```bash
# 새로운 가상환경 생성
conda create -n planty

# 새로운 가상환경 실행
conda activate planty

# 패키지 설치
pip install -r requirements.txt
```

- 파이썬 라이브러리 버전 충돌을 위해 가상환경을 세팅
- 그냥 python에서 `pip install -r requirements.txt`만 해도 동작 가능

</br>

## 🔐 API Key 등록 및 DB 연결

### 1. `.env` 파일 설정

`.env_example` 파일명을 `.env`로 변경하고 다음 항목을 설정하세요:

```env
# .env
COHERE_API_KEY = "your_cohere_api"
GROQ_API_KEY = "your_groq_api"
```

* [Cohere API Key 발급](https://dashboard.cohere.com/welcome/login?redirect_uri=%2Fapi-keys)
* [Groq API Key 발급](https://console.groq.com/keys)

### 2. HuggingFace 로그인 (로컬 모델 사용 시 필수)

```bash
huggingface-cli login
```

* [HuggingFace Token](https://huggingface.co/docs/hub/security-tokens)이 필요합니다.

### 3. MySQL DB 연결 정보

```json
# mysql 설정
{
  "host": "your_host_name.amazonaws.com",
  "port": 3306,
  "user": "your_user_name",
  "password": "your_password",
  "database": "your_database_name"
}
```

- `db_config_example.json` 파일의 이름을 `db_config.json`으로 변경
- 사용자의 데이터베이스 정보에 맞게 수정

</br>

## 🚀 서버 실행

### 1. FastAPI 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- `server_test.py`를 실행하여 서버 통신 테스트 가능

</br>

## 🤖 챗봇 설정

### `main.py` 사용 예시

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

# 1. 로컬 모델 사용 시
from chatbot_app import run_chatbot_with_ids

# 2. Groq API 사용 시 (아래 주석 해제)
# from groq_app import run_chatbot_with_ids
```

### 요청 형식

```json
POST /chat
Content-Type: application/json

{
  "chat_room_id": 1,
  "sensor_log_id": 10,
  "plant_env_standards_id": 5,
  "persona": "joy",
  "user_input": "식물이 괜찮은지 알려줘"
}
```

* `persona`는 다음 중 하나: `disgust`, `fear`, `joy`, `sadness`, `anger`

</br>

## 🌐 Groq API 사용

`groq_app.py` 내 모델 설정:

```python
lm = ChatGroq(
    model="gemma2-9b-it",  # 다른 Groq 모델로 변경 가능
    temperature=0.7,
    max_tokens=256,
)
```

> `main.py`에서 `chatbot_app` 대신 `groq_app`을 import 하면 Groq API를 통해 작동합니다.

</br>

## 📁 프로젝트 구조

```bash
Planty_Agent/
│
├── main.py                  # FastAPI 서버 진입점
├── chatbot_app.py           # 로컬 모델 기반 챗봇
├── groq_app.py              # Groq API 기반 챗봇
├── model_download.py        # 모델 다운로드 스크립트
├── requirements.txt         # 의존성 리스트
├── server_test.py           # 서버 통신 테스트
├── db_config.json           # 데이터베이스 정의 파일
├── .env                     # API Key 등 환경변수 파일 
└── HyperCLOVAX-Local/       # 로컬 모델 저장 디렉토리
```

</br>

## 🧪 참고 사항

* LangGraph, LangChain, HuggingFace Transformers 등을 기반으로 구성됨
* 식물 상태에 따른 맞춤형 응답 및 감정 기반 페르소나 설정 지원
* Chroma + Cohere 기반 문서 검색 및 Reranking 지원

