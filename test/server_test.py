# fastapi 서버 연결 테스트용 코드
import requests

def test_chat_api():
    url = "http://localhost:8000/chat"
    payload = {
        "type": "slm",
        "chat_room_id": 1,
        "sensor_log_id": 1,
        "plant_env_standards_id": 1,
        "persona": "joy",
        "user_input": "오늘 식물 상태가 어떤가요?"
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        print("챗봇 응답:", data.get("final_response"))
    else:
        print(f"오류 발생! 상태 코드: {response.status_code}, 메시지: {response.text}")
    
    url = "http://localhost:8000/plant_qa"
    payload_qa = {
        "type": "slm",
        "user_input": "몬스테라 키우는 방법을 알려줘"
    }

    response_qa = requests.post(url, json=payload_qa)
    if response_qa.status_code == 200:
        data_qa = response_qa.json()
        print("\n식물 QA 응답:", data_qa.get("final_response"))
    else:
        print(f"오류 발생! 상태 코드: {response_qa.status_code}, 메시지: {response_qa.text}")

if __name__ == "__main__":
    test_chat_api()
