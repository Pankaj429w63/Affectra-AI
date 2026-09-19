import os
import sys
import time
from fastapi.testclient import TestClient

# Ensure project root is in python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Set mock provider for validation
os.environ["LLM_PROVIDER"] = "mock"

from backend.app.main import app
from backend.app.services.rag_service import rag_service

def validate_rag_api():
    print("\n--- INITIALIZING RAG VALIDATION (TestClient) ---")
    
    # Pre-load retriever so we don't skew per-request timings too much
    rag_service.load()

    client = TestClient(app)

    test_queries = [
        "What emotions does Affectra AI recognize?",
        "What is the difference between emotion and sentiment?",
        "What are the limitations of the system?",
        "What model architecture does Affectra use?",
        # Unsupported question:
        "What is the capital of France?"
    ]
    
    all_passed = True

    for i, query in enumerate(test_queries):
        print(f"\n==================================================")
        print(f"QUERY {i+1}: {query}")
        print(f"==================================================")
        
        t0 = time.time()
        
        response = client.post(
            "/rag",
            json={"question": query, "top_k": 3}
        )
        
        t1 = time.time()
        print(f"Latency: {t1 - t0:.4f} seconds")
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"FAIL: HTTP Request failed: {response.text}")
            all_passed = False
            continue
            
        data = response.json()
        
        # Verify schema
        if "question" not in data or "answer" not in data or "retrieved_context" not in data:
            print("FAIL: Missing expected fields in response.")
            all_passed = False
            continue
            
        print(f"Answer: {data['answer']}")
        print(f"Context Items Returned: {len(data['retrieved_context'])}")
        
        # We expect context for valid Affectra queries, but we might get some for 'capital of France' too 
        # (just irrelevant ones because FAISS always returns the nearest ones)
        # But we still sanity check that something is returned
        if len(data['retrieved_context']) == 0:
            print("FAIL: No retrieved context returned.")
            all_passed = False
        else:
            for ctx in data['retrieved_context']:
                if not ctx.get('source') or ctx.get('chunk_index') is None or ctx.get('score') is None or not ctx.get('text'):
                    print("FAIL: Invalid context item.")
                    all_passed = False

    print("\n==================================================")
    print("--- RAG API SANITY CHECK REPORT ---")
    if all_passed:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
        sys.exit(1)

if __name__ == '__main__':
    validate_rag_api()
