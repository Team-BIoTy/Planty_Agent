# fastapi 서버 연결 테스트용 코드
import requests

def test_chat_api():
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

    chat_log = """안녕하세요! 오늘은 물을 잘 줬어요.
        기분이 어때요?

        오늘 햇빛이 많이 들어왔어요.
        그래서인지 잎이 더 반짝거려요.

        아침에는 조금 추웠는데, 지금은 따뜻해졌네요.
        혹시 오늘도 음악 틀어줄 수 있나요?"""


    url = "http://220.149.235.203:8002/chat_direct"
    payload = {
        "type": "llm",
        "nickname": "plan",     
        "env_info_dict": env_info,                      
        "cur_info_dict": cur_info,                     
        "chat_log": chat_log,
        "persona": "joy",
        "user_input": "오늘 기분이 어때?",
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        print("챗봇 응답:", data.get("final_response"))
    else:
        print(f"오류 발생! 상태 코드: {response.status_code}, 메시지: {response.text}")


    # url = "http://220.149.235.203:8002/chat"
    # payload = {
    #     "type": "llm",
    #     "chat_room_id": 1,
    #     "sensor_log_id": 1,
    #     "plant_env_standards_id": 1,
    #     "persona": "joy",
    #     "user_input": "오늘 식물 상태가 어떤가요?",
    # }

    # response = requests.post(url, json=payload)
    # if response.status_code == 200:
    #     data = response.json()
    #     print("챗봇 응답:", data.get("final_response"))
    # else:
    #     print(f"오류 발생! 상태 코드: {response.status_code}, 메시지: {response.text}")

    # url = "http://220.149.235.203:8002/plant_qa"
    # payload_qa = {
    #     "type": "llm",
    #     "user_input": "몬스테라 키우는 방법을 알려줘"
    # }

    # response_qa = requests.post(url, json=payload_qa)
    # if response_qa.status_code == 200:
    #     data_qa = response_qa.json()
    #     print("\n식물 QA 응답:", data_qa.get("final_response"))
    # else:
    #     print(f"오류 발생! 상태 코드: {response_qa.status_code}, 메시지: {response_qa.text}")

if __name__ == "__main__":
    test_chat_api()
