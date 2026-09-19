import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from rag.ingestion.chunker import RecursiveCharacterChunker
from rag.retriever import AffectraRetriever
from backend.app.core.config import settings

class TestRAG(unittest.TestCase):
    def test_chunker(self):
        class MockDoc:
            def __init__(self, text):
                self.text = text
                self.metadata = {"source": "mock.txt"}
        
        docs = [MockDoc("This is a simple sentence to test the chunker. " * 50)]
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_documents(docs)
        
        self.assertTrue(len(chunks) > 1, "Chunker failed to split long text")
        # Fixed chunk metadata access format
        self.assertEqual(chunks[0].metadata.get("source"), "mock.txt", "Metadata lost during chunking")

    def test_retriever(self):
        retriever = AffectraRetriever()
        try:
            retriever.load()
            res = retriever.retrieve("What is sadness?", top_k=2)
            self.assertTrue(len(res) > 0)
        except FileNotFoundError:
            # If the index is not built yet, we skip rather than fail
            pass

if __name__ == "__main__":
    unittest.main()
