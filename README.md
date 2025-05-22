# Planty_Agent

Planty Project에서 사용하는 LLM Agent입니다.

식물 정보와 페르소나를 반영하여 답변합니다. 

## 다운로드 

### git clone
```
git clone https://github.com/Team-BIoTy/Planty_Agent.git
cd Planty_agent
```
- `git clone`을 사용하여 코드 클론

### 모델 다운로드
- download_model.py

## 환경세팅
### 가상환경
```
# 새로운 가상환경 생성
conda create -n planty

# 새로운 가상환경 실행
conda activate planty

# 패키지 설치
pip install -r requirements.txt
```
- 새로운 가상환경을 생성한 뒤 실행에 필요한 패키지 설치

### API 설정 및 DB 연결
**API 설정**
```
# COHERE API 설정
COHERE_API_KEY = "your_cohere_api"
```
- `.env_example` 파일명을 `.env`로 수정
- your_cohere_api 자리에 cohere api key를 작성 후 저장 ([cohere api key 발급](https://dashboard.cohere.com/welcome/login?redirect_uri=%2Fapi-keys))

**DB 연결**
```
# mysql 설정
{
  "host": "your_host_name.amazonaws.com",
  "port": 3306,
  "user": "your_user_name",
  "password": "your_password",
  "database": "your_database_name"
}

```
- `db_test.json_example` 파일을 `db_test.json`으로 수정
- `db_test.json` 내용을 `mysql` 설정에 맞게 수정

## 서버 사용
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- powershell에서 unicorn을 사용하여 서버 운영