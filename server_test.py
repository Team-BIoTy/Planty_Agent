# fastapi 서버 연결 테스트용 코드
import requests

def test_chat_api():
    url = "http://localhost:8000/chat"
    payload = {
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

if __name__ == "__main__":
    test_chat_api()
