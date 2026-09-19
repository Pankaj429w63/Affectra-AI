import os
import sys
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_predict_raw_text():
    print("Testing /predict/raw with text input...")
    response = client.post(
        "/predict/raw",
        data={"text": "I am so happy and excited about this breakthrough!"}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    print("Text Prediction Result:")
    print("Emotion:", data["emotion"])
    print("Sentiment:", data["sentiment"])
    assert "label" in data["emotion"]
    assert "label" in data["sentiment"]
    assert "joy" in data["emotion"]["probabilities"]

def test_predict_raw_empty_fail():
    print("Testing /predict/raw with no inputs (should 400)...")
    response = client.post("/predict/raw")
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print("Empty test passed with 400 response.")

if __name__ == "__main__":
    test_predict_raw_text()
    test_predict_raw_empty_fail()
    print("\nALL PREDICT RAW TESTS PASSED SUCCESSFULLY!")
