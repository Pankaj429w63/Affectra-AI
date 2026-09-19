import os
import shutil
import unittest
import numpy as np
from pathlib import Path
from rag.vectorstore.faiss_store import FAISSStore

class TestFAISSStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_rag_vectorstore")
        self.store = FAISSStore(index_dir=str(self.test_dir))
        
        # Valid synthetic data
        self.valid_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)
        
        self.valid_metadata = [
            {"chunk_index": 0, "text": "Test 1"},
            {"chunk_index": 1, "text": "Test 2"}
        ]
        
    def tearDown(self):
        # Clean up test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def test_valid_embedding_matrix(self):
        self.store.build(self.valid_embeddings, self.valid_metadata)
        self.assertIsNotNone(self.store.index)
        self.assertEqual(self.store.size(), 2)
        self.assertEqual(self.store.dimension(), 3)
        
    def test_empty_embeddings_rejected(self):
        empty_embeddings = np.array([], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.store.build(empty_embeddings, [])
            
    def test_1d_embeddings_rejected(self):
        emb_1d = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.store.build(emb_1d, [{"text": "1D"}])
            
    def test_nan_embeddings_rejected(self):
        nan_embeddings = np.array([[0.1, np.nan, 0.3]], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.store.build(nan_embeddings, [{"text": "NaN"}])
            
    def test_infinite_embeddings_rejected(self):
        inf_embeddings = np.array([[0.1, np.inf, 0.3]], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.store.build(inf_embeddings, [{"text": "Inf"}])
            
    def test_correct_faiss_dimension_and_count(self):
        self.store.build(self.valid_embeddings, self.valid_metadata)
        self.assertEqual(self.store.dimension(), 3)
        self.assertEqual(self.store.size(), 2)
        
    def test_metadata_count_mismatch(self):
        # 2 embeddings but 1 metadata
        with self.assertRaises(ValueError):
            self.store.build(self.valid_embeddings, [{"text": "Only one"}])
            
    def test_save_load_functionality(self):
        self.store.build(self.valid_embeddings, self.valid_metadata)
        self.store.save()
        
        self.assertTrue(self.store.index_path.exists())
        self.assertTrue(self.store.meta_path.exists())
        
        new_store = FAISSStore(index_dir=str(self.test_dir))
        new_store.load()
        
        self.assertEqual(new_store.size(), 2)
        self.assertEqual(new_store.dimension(), 3)
        self.assertEqual(len(new_store.metadata), 2)
        self.assertEqual(new_store.metadata[0]["text"], "Test 1")
        
    def test_deterministic_metadata_ordering(self):
        self.store.build(self.valid_embeddings, self.valid_metadata)
        # Verify the order of metadata lists remains untouched
        self.assertEqual(self.store.metadata[0]["chunk_index"], 0)
        self.assertEqual(self.store.metadata[1]["chunk_index"], 1)

if __name__ == '__main__':
    unittest.main()
