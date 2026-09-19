import os
import sys
import time
from fastapi.testclient import TestClient

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Set mock provider for validation
os.environ["LLM_PROVIDER"] = "mock"

from backend.app.main import app
from backend.app.services.rag_service import rag_service

def validate_complete_rag():
    print("\n--- INITIALIZING COMPLETE RAG VALIDATION ---")
    
    # Pre-load retriever
    rag_service.load()

    client = TestClient(app)

    test_queries = [
        "What emotions does Affectra AI recognize?",
        "What sentiment labels does Affectra AI use?",
        "What is the difference between emotion and sentiment?",
        "What are the limitations of Affectra AI?",
        "What model architecture does Affectra AI use?",
        # Unsupported question:
        "What is the capital of France?"
    ]
    
    all_passed = True

    print("\n==================================================")
    print("1. TEST API ENDPOINTS /rag VALIDATION")
    print("==================================================")
    
    # top_k=1
    res = client.post("/rag", json={"question": "test", "top_k": 1})
    if res.status_code == 200:
        print("top_k=1: PASS")
    else:
        print(f"top_k=1: FAIL ({res.status_code})")
        all_passed = False
        
    # top_k=3
    res = client.post("/rag", json={"question": "test", "top_k": 3})
    if res.status_code == 200:
        print("top_k=3: PASS")
    else:
        print(f"top_k=3: FAIL ({res.status_code})")
        all_passed = False
        
    # top_k=7
    res = client.post("/rag", json={"question": "test", "top_k": 7})
    if res.status_code == 200:
        print("top_k=7: PASS")
    else:
        print(f"top_k=7: FAIL ({res.status_code})")
        all_passed = False
        
    # top_k greater than available chunks (e.g., 100)
    res = client.post("/rag", json={"question": "test", "top_k": 100})
    if res.status_code == 200:
        print("top_k=100 (greater than chunks): PASS")
    else:
        print(f"top_k=100 (greater than chunks): FAIL ({res.status_code})")
        all_passed = False

    # empty question
    res = client.post("/rag", json={"question": "", "top_k": 3})
    if res.status_code == 422:
        print("empty question: PASS")
    else:
        print(f"empty question: FAIL ({res.status_code})")
        all_passed = False

    # whitespace question
    res = client.post("/rag", json={"question": "   \n\t", "top_k": 3})
    if res.status_code == 422:
        print("whitespace question: PASS")
    else:
        print(f"whitespace question: FAIL ({res.status_code})")
        all_passed = False

    # invalid top_k
    res = client.post("/rag", json={"question": "test", "top_k": 0})
    if res.status_code == 422:
        print("invalid top_k (0): PASS")
    else:
        print(f"invalid top_k (0): FAIL ({res.status_code})")
        all_passed = False

    # malformed request
    res = client.post("/rag", json={"q": "test"})
    if res.status_code == 422:
        print("malformed request: PASS")
    else:
        print(f"malformed request: FAIL ({res.status_code})")
        all_passed = False


    print("\n==================================================")
    print("2. TEST REAL KNOWLEDGE BASE QUERIES")
    print("==================================================")
    for i, query in enumerate(test_queries):
        print(f"\nQUERY {i+1}: {query}")
        
        t0 = time.time()
        response = client.post("/rag", json={"question": query, "top_k": 3})
        t1 = time.time()
        
        if response.status_code != 200:
            print(f"FAIL: HTTP Request failed: {response.text}")
            all_passed = False
            continue
            
        data = response.json()
        print(f"Latency: {t1 - t0:.4f} seconds")
        print(f"Answer: {data.get('answer', 'MISSING')[:100]}...")
        
        ctxs = data.get('retrieved_context', [])
        print(f"Context Items Returned: {len(ctxs)}")
        
        if len(ctxs) == 0:
            print("FAIL: No retrieved context returned.")
            all_passed = False
        else:
            prev_score = float('inf')
            for j, ctx in enumerate(ctxs):
                src = ctx.get('source')
                idx = ctx.get('chunk_index')
                sc = ctx.get('score')
                txt = ctx.get('text')
                
                if not src or idx is None or sc is None or not txt:
                    print(f"  -> FAIL: Item {j} has invalid schema.")
                    all_passed = False
                
                if sc > prev_score + 1e-5:
                    print(f"  -> FAIL: Scores are not descending! {sc} > {prev_score}")
                    all_passed = False
                prev_score = sc

    print("\n==================================================")
    print("3. TEST OTHER ENDPOINTS")
    print("==================================================")
    res = client.get("/health")
    print(f"GET /health: {res.status_code}")
    if res.status_code != 200: all_passed = False
    
    # We will test /docs
    res = client.get("/docs")
    print(f"GET /docs: {res.status_code}")
    if res.status_code != 200: all_passed = False

    print("\n==================================================")
    print("--- COMPLETE RAG API VALIDATION REPORT ---")
    if all_passed:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
        sys.exit(1)

if __name__ == '__main__':
    validate_complete_rag()
