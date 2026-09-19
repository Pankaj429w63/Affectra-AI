import os
import tempfile
import torch
from typing import Optional, List
from training.src.utils import get_logger

logger = get_logger(__name__)

class FeatureExtractionService:
    _instance: Optional["FeatureExtractionService"] = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.text_extractor = None
        self.audio_extractor = None
        self.video_extractor = None

    @classmethod
    def get_instance(cls) -> "FeatureExtractionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_text_extractor(self):
        if self.text_extractor is None:
            from training.src.feature_extractors import TextExtractor
            logger.info("Initializing TextExtractor (distilroberta-base)...")
            self.text_extractor = TextExtractor(device=self.device)
        return self.text_extractor

    def _get_audio_extractor(self):
        if self.audio_extractor is None:
            from training.src.feature_extractors import AudioExtractor
            logger.info("Initializing AudioExtractor (facebook/wav2vec2-base)...")
            self.audio_extractor = AudioExtractor(device=self.device)
        return self.audio_extractor

    def _get_video_extractor(self):
        if self.video_extractor is None:
            from training.src.feature_extractors import VideoExtractor
            logger.info("Initializing VideoExtractor (google/vit-base-patch16-224)...")
            self.video_extractor = VideoExtractor(device=self.device)
        return self.video_extractor

    def extract_text_feature(self, text: Optional[str]) -> List[float]:
        if not text or not text.strip():
            return [0.0] * 768
        extractor = self._get_text_extractor()
        tensor = extractor.extract_batch([text.strip()])
        return tensor[0].tolist()

    def extract_audio_feature(self, file_bytes: Optional[bytes], filename: Optional[str] = None) -> List[float]:
        if not file_bytes or len(file_bytes) == 0:
            return [0.0] * 768

        suffix = os.path.splitext(filename)[1] if filename else ".wav"
        if not suffix:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            extractor = self._get_audio_extractor()
            tensor = extractor.extract_single(tmp_path)
            return tensor.tolist()
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return [0.0] * 768
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def extract_video_feature(self, file_bytes: Optional[bytes], filename: Optional[str] = None) -> List[float]:
        if not file_bytes or len(file_bytes) == 0:
            return [0.0] * 768

        suffix = os.path.splitext(filename)[1] if filename else ".mp4"
        if not suffix:
            suffix = ".mp4"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            extractor = self._get_video_extractor()
            tensor = extractor.extract_single(tmp_path)
            return tensor.tolist()
        except Exception as e:
            logger.error(f"Video feature extraction failed: {e}")
            return [0.0] * 768
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

feature_extraction_service = FeatureExtractionService.get_instance()
