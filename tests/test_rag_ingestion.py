import os
import tempfile
import unittest
from pathlib import Path
from rag.ingestion.loader import DirectoryLoader, Document
from rag.ingestion.chunker import RecursiveCharacterChunker, Chunk

class TestDirectoryLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        
    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_markdown_loads(self):
        file_path = self.dir_path / "test1.md"
        file_path.write_text("Hello Markdown", encoding="utf-8")
        loader = DirectoryLoader(str(self.dir_path))
        docs = loader.load()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "Hello Markdown")
        self.assertEqual(docs[0].metadata["source"], "test1.md")

    def test_valid_txt_loads(self):
        file_path = self.dir_path / "test2.txt"
        file_path.write_text("Hello TXT", encoding="utf-8")
        loader = DirectoryLoader(str(self.dir_path))
        docs = loader.load()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "Hello TXT")
        self.assertEqual(docs[0].metadata["source"], "test2.txt")

    def test_multiple_documents_load(self):
        (self.dir_path / "1.md").write_text("One", encoding="utf-8")
        (self.dir_path / "2.txt").write_text("Two", encoding="utf-8")
        loader = DirectoryLoader(str(self.dir_path))
        docs = loader.load()
        self.assertEqual(len(docs), 2)

    def test_empty_document_handling(self):
        (self.dir_path / "empty.md").write_text("", encoding="utf-8")
        loader = DirectoryLoader(str(self.dir_path))
        docs = loader.load()
        self.assertEqual(len(docs), 0)

    def test_missing_directory(self):
        loader = DirectoryLoader(str(self.dir_path / "does_not_exist"))
        with self.assertRaises(FileNotFoundError):
            loader.load()

    def test_unsupported_extension(self):
        (self.dir_path / "test.jpg").write_bytes(b"fake_image_data")
        loader = DirectoryLoader(str(self.dir_path))
        docs = loader.load()
        self.assertEqual(len(docs), 0)

    def test_utf8_text_works(self):
        (self.dir_path / "utf8.md").write_text("Hello 🌍", encoding="utf-8")
        loader = DirectoryLoader(str(self.dir_path))
        docs = loader.load()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "Hello 🌍")

class TestRecursiveCharacterChunker(unittest.TestCase):
    def test_short_document_one_chunk(self):
        doc = Document("Short text", {"source": "test.md"})
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split_documents([doc])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short text")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[0].metadata["source"], "test.md")

    def test_long_document_multiple_chunks(self):
        doc = Document("A" * 1000, {"source": "test.md"})
        chunker = RecursiveCharacterChunker(chunk_size=400, chunk_overlap=50)
        chunks = chunker.split_documents([doc])
        self.assertTrue(len(chunks) > 1)

    def test_no_empty_chunk(self):
        doc = Document("   \n   \n", {"source": "test.md"})
        chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.split_documents([doc])
        self.assertEqual(len(chunks), 0)

    def test_chunk_indexes_sequential(self):
        doc = Document("A" * 1000, {"source": "test.md"})
        chunker = RecursiveCharacterChunker(chunk_size=400, chunk_overlap=50)
        chunks = chunker.split_documents([doc])
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata["chunk_index"], i)

    def test_chunk_overlap_behaves_as_configured(self):
        doc = Document("1234567890", {"source": "test.md"})
        # 10 chars. chunk_size=6, overlap=2
        # chunk 0: 123456
        # chunk 1: 567890
        chunker = RecursiveCharacterChunker(chunk_size=6, chunk_overlap=2)
        chunks = chunker.split_documents([doc])
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "123456")
        self.assertEqual(chunks[1].text, "567890")
        self.assertEqual(chunks[2].text, "90")

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            RecursiveCharacterChunker(chunk_size=0)
        with self.assertRaises(ValueError):
            RecursiveCharacterChunker(chunk_size=-10)

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=-10)

    def test_overlap_greater_than_or_equal_to_chunk_size(self):
        with self.assertRaises(ValueError):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=150)

    def test_chunking_terminates_correctly(self):
        doc = Document("word " * 100, {"source": "test.md"})
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=20)
        chunks = chunker.split_documents([doc])
        # If it doesn't infinite loop, it terminates.
        self.assertTrue(len(chunks) > 0)

if __name__ == '__main__':
    unittest.main()
