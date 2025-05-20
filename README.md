# Planty_Agent

Planty Project에서 사용하는 LLM Agent입니다.

식물 정보와 페르소나를 반영하여 답변합니다. 

## 사용법

```
git clone https://github.com/Team-BIoTy/Planty_Agent.git
cd Planty_agent
```
- `git clone`을 사용하여 코드 클론

```
# 모델 다운로드 
huggingface-cli download yerim00/HyperCLOVAX-SEED-Text-Instruct-1.5B-planty-ia3 --local-dir HyperCLOVAX-Local
```
- 위의 코드를 실행하여 모델을 HyperCLOVAX-Local 폴더에 다운로드

```
# 새로운 가상환경
conda create -n planty

# 패키지 설치
pip install -r requirements.txt
```
- 새로운 가상환경을 생성한 뒤 실행에 필요한 패키지 설치

```
# COHERE API 설정
COHERE_API_KEY = "your_cohere_api"
```
- `.env_example` 파일명을 `.env`로 수정
- your_cohere_api 자리에 cohere api key를 작성 후 저장 ([cohere api key 발급](https://dashboard.cohere.com/welcome/login?redirect_uri=%2Fapi-keys))

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