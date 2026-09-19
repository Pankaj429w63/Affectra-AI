import os
import sys
import torch
import json
import urllib.request
from urllib.error import HTTPError, URLError

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from training.src.inference_model import AffectraPredictor

def main():
    print("="*50)
    print("PHASE 2.1 — REAL MODEL INTEGRATION & API VALIDATION")
    print("="*50)
    
    # STEP 2: Locate Real Feature Data
    print("\n[1] Checking Feature Data")
    text_path = os.path.join(project_root, "data/feature_cache/text_test.pt")
    audio_path = os.path.join(project_root, "data/feature_cache/audio_test.pt")
    video_path = os.path.join(project_root, "data/feature_cache/video_test.pt")
    
    text_test = torch.load(text_path, weights_only=True)
    audio_test = torch.load(audio_path, weights_only=True)
    video_test = torch.load(video_path, weights_only=True)
    
    print(f"text_test shape: {text_test.shape}")
    print(f"audio_test shape: {audio_test.shape}")
    print(f"video_test shape: {video_test.shape}")
    
    assert text_test.shape == (2610, 768), "Unexpected text shape"
    assert audio_test.shape == (2610, 768), "Unexpected audio shape"
    assert video_test.shape == (2610, 768), "Unexpected video shape"
    
    text_nan = torch.isnan(text_test).any().item()
    audio_nan = torch.isnan(audio_test).any().item()
    video_nan = torch.isnan(video_test).any().item()
    print(f"NaN check - Text: {text_nan}, Audio: {audio_nan}, Video: {video_nan}")
    
    # STEP 3: Small Real-Data API Test (Direct Predictor)
    print("\n[2] Direct Predictor Test (Sample 0)")
    model_dir = os.path.join(project_root, "models/affectra_multimodal")
    predictor = AffectraPredictor(model_dir=model_dir, device="cpu")
    
    t_feat = text_test[0:1]
    a_feat = audio_test[0:1]
    v_feat = video_test[0:1]
    
    out = predictor.predict(t_feat, a_feat, v_feat)
    
    print(f"Emotion label: {out['emotion']['labels'][0]}")
    print("Emotion probabilities:", {k: v.item() for k, v in zip(predictor.emotion_labels.values(), out['emotion']['probabilities'][0])})
    
    print(f"Sentiment label: {out['sentiment']['labels'][0]}")
    print("Sentiment probabilities:", {k: v.item() for k, v in zip(predictor.sentiment_labels.values(), out['sentiment']['probabilities'][0])})
    
    assert len(out['emotion']['probabilities'][0]) == 7
    assert len(out['sentiment']['probabilities'][0]) == 3
    assert abs(out['emotion']['probabilities'][0].sum().item() - 1.0) < 1e-5
    assert abs(out['sentiment']['probabilities'][0].sum().item() - 1.0) < 1e-5
    
    # STEP 4 & 5: Test through Actual FastAPI
    print("\n[3] Testing FastAPI Endpoints")
    base_url = "http://127.0.0.1:8000"
    
    try:
        health_res = json.loads(urllib.request.urlopen(f"{base_url}/health").read())
        print(f"GET /health: {health_res}")
    except Exception as e:
        print(f"Failed to reach FastAPI /health: {e}")
        return

    # Test multiple samples
    print("\n[4] Testing Multiple Real Samples via API")
    num_samples = 5
    for i in range(num_samples):
        req_data = {
            "text_feat": text_test[i].tolist(),
            "audio_feat": audio_test[i].tolist(),
            "video_feat": video_test[i].tolist()
        }
        
        data = json.dumps(req_data).encode('utf-8')
        req = urllib.request.Request(f"{base_url}/predict", data=data, headers={'Content-Type': 'application/json'})
        
        res = json.loads(urllib.request.urlopen(req).read())
        e_label = res['emotion']['label']
        s_label = res['sentiment']['label']
        
        e_sum = sum(res['emotion']['probabilities'].values())
        s_sum = sum(res['sentiment']['probabilities'].values())
        
        print(f"Sample {i} -> Emotion: {e_label} (sum={e_sum:.4f}) | Sentiment: {s_label} (sum={s_sum:.4f})")
        assert abs(e_sum - 1.0) < 1e-4
        assert abs(s_sum - 1.0) < 1e-4
    
    # STEP 6: Error Handling
    print("\n[5] Testing Error Handling")
    
    def expect_error(data_dict, description, expected_code=422):
        try:
            req_data = json.dumps(data_dict).encode('utf-8')
            req = urllib.request.Request(f"{base_url}/predict", data=req_data, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req)
            print(f"FAIL: {description} did not raise an error.")
        except HTTPError as e:
            if e.code == expected_code:
                print(f"PASS: {description} -> HTTP {e.code}")
            else:
                print(f"FAIL: {description} -> Expected HTTP {expected_code}, got HTTP {e.code}")
            
    # 1. Missing modality
    expect_error({
        "text_feat": text_test[0].tolist(),
        "audio_feat": audio_test[0].tolist()
    }, "Missing video_feat")
    
    # 2. Wrong feature size
    expect_error({
        "text_feat": text_test[0].tolist()[:100], # wrong length
        "audio_feat": audio_test[0].tolist(),
        "video_feat": video_test[0].tolist()
    }, "Wrong feature size")
    
    # 3. Invalid JSON type (string instead of list)
    expect_error({
        "text_feat": "not a list",
        "audio_feat": audio_test[0].tolist(),
        "video_feat": video_test[0].tolist()
    }, "Invalid JSON type")

if __name__ == "__main__":
    main()
