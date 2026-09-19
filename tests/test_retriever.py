import os
import shutil
import unittest
import numpy as np
from pathlib import Path
from rag.retriever import AffectraRetriever, RetrievalResult
from rag.vectorstore.faiss_store import FAISSStore
from rag.ingestion.embeddings import EmbeddingModel

class TestAffectraRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a tiny synthetic index for fast, isolated tests
        cls.test_dir = Path("data/test_rag_retriever")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the real embedding model ONCE to generate real valid embeddings for the dummy data
        # so that FAISS search doesn't fail with dimension mismatches or nonsense inner products
        embedder = EmbeddingModel(device="cpu")
        dim = embedder.dimension
        
        # Create synthetic valid embeddings
        # Vectors will just be random normal normalized
        cls.synth_embeddings = np.random.randn(5, dim).astype(np.float32)
        row_norms = np.linalg.norm(cls.synth_embeddings, axis=1, keepdims=True)
        cls.synth_embeddings = cls.synth_embeddings / row_norms
        
        cls.synth_metadata = [
            {"chunk_index": i, "text": f"Test document {i}", "source": "test.md"} for i in range(5)
        ]
        
        # Build and save FAISS store
        store = FAISSStore(index_dir=str(cls.test_dir))
        store.build(cls.synth_embeddings, cls.synth_metadata)
        store.save()

    @classmethod
    def tearDownClass(cls):
        # Clean up
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def setUp(self):
        # Load retriever for each test
        self.retriever = AffectraRetriever(index_dir=str(self.test_dir))
        self.retriever.load()
        
    def test_valid_query(self):
        results = self.retriever.retrieve("What is this?", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievalResult)
        
    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve("")
            
    def test_whitespace_query_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve("   \n \t  ")
            
    def test_invalid_query_type_rejected(self):
        with self.assertRaises(TypeError):
            self.retriever.retrieve(["not a string"])
            
    def test_top_k_1(self):
        results = self.retriever.retrieve("test", top_k=1)
        self.assertEqual(len(results), 1)
        
    def test_top_k_3(self):
        results = self.retriever.retrieve("test", top_k=3)
        self.assertEqual(len(results), 3)
        
    def test_top_k_greater_than_index_size(self):
        # Index size is 5
        results = self.retriever.retrieve("test", top_k=10)
        self.assertEqual(len(results), 5)
        
    def test_invalid_top_k_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", top_k=0)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", top_k=-5)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", top_k="five")
            
    def test_result_contents(self):
        results = self.retriever.retrieve("test", top_k=1)
        res = results[0]
        self.assertTrue(res.text.startswith("Test document"))
        self.assertEqual(res.source, "test.md")
        self.assertIn(res.chunk_index, [0, 1, 2, 3, 4])
        self.assertIsNotNone(res.score)
        
        import math
        self.assertFalse(math.isnan(res.score))
        self.assertFalse(math.isinf(res.score))
        
    def test_results_sorted_descending(self):
        results = self.retriever.retrieve("test", top_k=5)
        for i in range(len(results) - 1):
            self.assertTrue(results[i].score >= results[i+1].score)
            
    def test_faiss_id_maps_to_metadata(self):
        # Even though we don't control exact FAISS IDs returned here easily,
        # we know the text should strictly contain the chunk index.
        results = self.retriever.retrieve("test mapping", top_k=5)
        for res in results:
            expected_text = f"Test document {res.chunk_index}"
            self.assertEqual(res.text, expected_text)
            
    def test_retriever_load_from_disk(self):
        # This was already technically done in setUp, but let's be explicit
        fresh_retriever = AffectraRetriever(index_dir=str(self.test_dir))
        self.assertFalse(fresh_retriever.is_loaded)
        fresh_retriever.load()
        self.assertTrue(fresh_retriever.is_loaded)
        
        # Ensure it works after loading
        results = fresh_retriever.retrieve("disk load test", top_k=1)
        self.assertEqual(len(results), 1)

if __name__ == '__main__':
    unittest.main()
