import json
import urllib.request
from urllib.error import HTTPError

def main():
    print("="*50)
    print("PHASE 3 — LLM EXPLANATION TEST")
    print("="*50)

    base_url = "http://127.0.0.1:8000"

    print("\n[1] Testing POST /explain with Mock Provider")
    req_data = {
        "emotion_label": "joy",
        "emotion_probabilities": {
            "anger": 0.05,
            "disgust": 0.02,
            "fear": 0.01,
            "joy": 0.85,
            "neutral": 0.05,
            "sadness": 0.01,
            "surprise": 0.01
        },
        "sentiment_label": "positive",
        "sentiment_probabilities": {
            "positive": 0.90,
            "negative": 0.05,
            "neutral": 0.05
        }
    }
    
    data = json.dumps(req_data).encode('utf-8')
    req = urllib.request.Request(f"{base_url}/explain", data=data, headers={'Content-Type': 'application/json'})
    
    try:
        res = json.loads(urllib.request.urlopen(req).read())
        print("PASS: /explain returned successfully.")
        print(f"Explanation Output: {res['explanation']}")
    except HTTPError as e:
        print(f"FAIL: Expected 200 OK, got HTTP {e.code}")

    print("\n[2] Testing POST /explain Error Handling (Missing Fields)")
    req_data_invalid = {
        "emotion_label": "joy"
    }
    data_inv = json.dumps(req_data_invalid).encode('utf-8')
    req_inv = urllib.request.Request(f"{base_url}/explain", data=data_inv, headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req_inv)
        print("FAIL: Expected 422 for missing fields")
    except HTTPError as e:
        if e.code == 422:
            print(f"PASS: Handled invalid request correctly (HTTP {e.code})")
        else:
            print(f"FAIL: Expected 422, got HTTP {e.code}")

if __name__ == "__main__":
    main()
