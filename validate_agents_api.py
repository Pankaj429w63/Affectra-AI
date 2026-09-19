import os
from fastapi.testclient import TestClient
from backend.app.main import app

def test_api():
    os.environ["LLM_PROVIDER"] = "mock"
    client = TestClient(app)
    agents_url = "/agents"

    print("==================================================")
    print("1. Emotion + Sentiment + RAG")
    print("==================================================")
    payload_1 = {
        "emotion_label": "joy",
        "emotion_probabilities": {"joy": 0.8, "sadness": 0.2},
        "sentiment_label": "positive",
        "sentiment_probabilities": {"positive": 0.9, "negative": 0.1},
        "user_query": "What emotions does Affectra AI recognize?"
    }
    response = client.post(agents_url, json=payload_1)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Final Response:", data.get("final_response"))
        print("Safety Approved:", data.get("safety_approved"))
        print("Trace:", data.get("execution_trace"))
        print("Retrieved context items:", len(data.get("retrieved_context", [])))
    else:
        print("Response:", response.text)

    print("\n==================================================")
    print("2. Different Emotion + Sentiment (No Query)")
    print("==================================================")
    payload_2 = {
        "emotion_label": "anger",
        "emotion_probabilities": {"anger": 0.75, "neutral": 0.25},
        "sentiment_label": "negative",
        "sentiment_probabilities": {"negative": 0.85, "neutral": 0.15}
    }
    response = client.post(agents_url, json=payload_2)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Final Response:", data.get("final_response"))
        print("Safety Approved:", data.get("safety_approved"))
        print("Trace:", data.get("execution_trace"))
    else:
        print("Response:", response.text)

    print("\n==================================================")
    print("3. Neutral Emotion + Neutral Sentiment")
    print("==================================================")
    payload_3 = {
        "emotion_label": "neutral",
        "sentiment_label": "neutral"
    }
    response = client.post(agents_url, json=payload_3)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Final Response:", data.get("final_response"))
        print("Trace:", data.get("execution_trace"))
    else:
        print("Response:", response.text)

    print("\n==================================================")
    print("4. Pure Query (No ML Prediction)")
    print("==================================================")
    payload_4 = {
        "user_query": "Explain sentiment analysis."
    }
    response = client.post(agents_url, json=payload_4)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Final Response:", data.get("final_response"))
        print("Trace:", data.get("execution_trace"))
    else:
        print("Response:", response.text)

if __name__ == "__main__":
    test_api()

if __name__ == "__main__":
    test_api()
