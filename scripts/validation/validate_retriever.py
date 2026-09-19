import os
import sys
import time

# Ensure project root is in python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from rag.retriever import AffectraRetriever

def validate_retriever():
    print("\n--- INITIALIZING RETRIEVER ---")
    
    t0 = time.time()
    try:
        retriever = AffectraRetriever()
        retriever.load()
    except Exception as e:
        print(f"Failed to load retriever: {e}")
        sys.exit(1)
    t1 = time.time()
    init_time = t1 - t0
    print(f"Retriever initialization & FAISS load time: {init_time:.4f} seconds")
    
    # Meaningful queries based on the emotion knowledge base
    test_queries = [
        "What emotions does Affectra AI recognize?",
        "What is the difference between sentiment and emotion?",
        "Are there any limitations with Affectra?",
        "Tell me about the multimodal architecture"
    ]
    
    all_passed = True
    
    for i, query in enumerate(test_queries):
        print(f"\n==================================================")
        print(f"QUERY {i+1}: {query}")
        print(f"==================================================")
        
        t_query_start = time.time()
        
        # We retrieve top 3
        try:
            results = retriever.retrieve(query, top_k=3)
        except Exception as e:
            print(f"Retrieval failed for query '{query}': {e}")
            all_passed = False
            continue
            
        t_query_end = time.time()
        print(f"Retrieval Time: {t_query_end - t_query_start:.4f} seconds")
        print(f"Results returned: {len(results)}\n")
        
        # Sanity check variables
        if len(results) == 0:
            print("FAIL: No results returned.")
            all_passed = False
            continue
            
        last_score = float('inf')
        
        for j, res in enumerate(results):
            print(f"--- Rank {j+1} (Score: {res.score:.4f}) ---")
            print(f"Source: {res.source} | Chunk Index: {res.chunk_index}")
            print(f"Text Snippet: {res.text[:150]}...\n")
            
            # Validations
            if not res.text:
                print("FAIL: Result text is empty.")
                all_passed = False
            if not res.source:
                print("FAIL: Result source is missing.")
                all_passed = False
            if res.chunk_index < 0:
                print("FAIL: Invalid chunk index.")
                all_passed = False
            import math
            if math.isnan(res.score) or math.isinf(res.score):
                print("FAIL: Score is not finite.")
                all_passed = False
            if res.score > last_score + 1e-5: # adding small epsilon for float precision
                print(f"FAIL: Results are not ordered descending ({res.score} > {last_score}).")
                all_passed = False
                
            last_score = res.score
            
    print("\n==================================================")
    print("--- RETRIEVAL SANITY CHECK REPORT ---")
    print(f"Initialization Time: {init_time:.4f} s")
    if all_passed:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
        sys.exit(1)

if __name__ == '__main__':
    validate_retriever()
