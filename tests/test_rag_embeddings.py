import unittest
import numpy as np
from rag.ingestion.embeddings import EmbeddingModel
from rag.ingestion.chunker import Chunk

class TestEmbeddingModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load model once for all tests to keep tests fast
        cls.embedder = EmbeddingModel(device="cpu")
        
    def test_model_loads_and_device(self):
        self.assertIsNotNone(self.embedder.model)
        self.assertEqual(self.embedder.device, "cpu")
        self.assertTrue(self.embedder.dimension > 0)
        
    def test_single_text_produces_one_embedding(self):
        chunk = Chunk(text="Hello world", metadata={"source": "test.txt", "chunk_index": 0})
        embeddings = self.embedder.embed_chunks([chunk])
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), self.embedder.dimension)
        
    def test_multiple_texts_produce_matching_embeddings(self):
        chunks = [
            Chunk(text="Hello world"),
            Chunk(text="This is a test"),
            Chunk(text="Another text chunk")
        ]
        embeddings = self.embedder.embed_chunks(chunks)
        self.assertEqual(len(embeddings), 3)
        for emb in embeddings:
            self.assertEqual(len(emb), self.embedder.dimension)
            
    def test_empty_input_handled_correctly(self):
        embeddings = self.embedder.embed_chunks([])
        self.assertEqual(len(embeddings), 0)
        
    def test_no_nans_or_infs(self):
        chunks = [Chunk(text="Testing mathematical validity.")]
        embeddings = self.embedder.embed_chunks(chunks)
        emb_np = np.array(embeddings)
        self.assertEqual(np.isnan(emb_np).sum(), 0)
        self.assertEqual(np.isinf(emb_np).sum(), 0)
        
    def test_normalized_output_is_numerically_valid(self):
        chunks = [Chunk(text="Normalize me.")]
        embeddings = self.embedder.embed_chunks(chunks)
        emb_np = np.array(embeddings[0])
        # Vector norm should be approximately 1.0
        norm = np.linalg.norm(emb_np)
        self.assertAlmostEqual(norm, 1.0, places=4)
        
    def test_deterministic_behavior(self):
        chunks = [Chunk(text="Deterministic test.")]
        emb1 = self.embedder.embed_chunks(chunks)
        emb2 = self.embedder.embed_chunks(chunks)
        
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5, atol=1e-5)
        
    def test_metadata_not_mutated(self):
        meta = {"source": "test.txt", "chunk_index": 42}
        chunk = Chunk(text="Test chunk", metadata=meta)
        
        self.embedder.embed_chunks([chunk])
        
        # Ensure the chunk itself and metadata weren't modified by the embedding process
        self.assertEqual(chunk.metadata["source"], "test.txt")
        self.assertEqual(chunk.metadata["chunk_index"], 42)

if __name__ == '__main__':
    unittest.main()
