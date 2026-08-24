"""
Affectra AI — Pretrained Feature Extractors
=============================================
Three frozen encoder classes that convert raw MELD data into
fixed-size feature vectors:

  TextExtractor:   distilroberta-base  → [1, 768] via [CLS] token
  AudioExtractor:  facebook/wav2vec2-base → [1, 768] via attention-mask mean-pool
  VideoExtractor:  google/vit-base-patch16-224 → [1, 768] via 8-frame mean-pool

IMPORTANT:
  - All encoders are frozen (requires_grad=False) before use.
  - Extraction runs once per dataset; results are saved to the feature cache.
  - Audio is extracted from each .mp4 via FFmpeg (subprocess).
  - Video frames are sampled uniformly from each .mp4 via OpenCV.
  - Missing/corrupt files produce zero vectors (never crash the pipeline).
"""

import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    ViTImageProcessor,
    ViTModel,
)

from training.src.config import (
    AUDIO_ENCODER_NAME,
    AUDIO_FEATURE_DIM,
    AUDIO_SAMPLE_RATE,
    TEXT_ENCODER_NAME,
    TEXT_FEATURE_DIM,
    TEXT_MAX_LENGTH,
    VIDEO_ENCODER_NAME,
    VIDEO_FEATURE_DIM,
    VIDEO_FRAMES_PER_CLIP,
)
from training.src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base class helpers
# ---------------------------------------------------------------------------

def _freeze_model(model: torch.nn.Module) -> None:
    """Freeze all parameters — encoders must not be trained."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


# ---------------------------------------------------------------------------
# Text Extractor — DistilRoBERTa
# ---------------------------------------------------------------------------

class TextExtractor:
    """
    Extracts [CLS] token embeddings from DistilRoBERTa for a list of strings.

    Output shape per sample: [768]
    """

    def __init__(self, device: torch.device, batch_size: int = 64):
        logger.info(f"Loading text encoder: {TEXT_ENCODER_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)
        self.model = AutoModel.from_pretrained(TEXT_ENCODER_NAME).to(device)
        _freeze_model(self.model)
        self.device = device
        self.batch_size = batch_size
        logger.info(f"  Text encoder ready on {device}")

    @torch.no_grad()
    def extract_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Extract CLS embeddings for a list of text strings.

        Args:
            texts: List of N utterance strings.

        Returns:
            FloatTensor of shape [N, 768].
        """
        all_features: List[torch.Tensor] = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Text extraction"):
            batch_texts = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=TEXT_MAX_LENGTH,
                padding="max_length",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)

            # [CLS] token is at position 0 of last_hidden_state
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 768]
            all_features.append(cls_embeddings.cpu())

        return torch.cat(all_features, dim=0)  # [N, 768]

    def extract_all_splits(
        self,
        train_texts: List[str],
        dev_texts: List[str],
        test_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features for all three splits."""
        logger.info("Extracting text features for TRAIN split...")
        train_feats = self.extract_batch(train_texts)

        logger.info("Extracting text features for DEV split...")
        dev_feats = self.extract_batch(dev_texts)

        logger.info("Extracting text features for TEST split...")
        test_feats = self.extract_batch(test_texts)

        logger.info(
            f"Text features: train={train_feats.shape}, "
            f"dev={dev_feats.shape}, test={test_feats.shape}"
        )
        return train_feats, dev_feats, test_feats


# ---------------------------------------------------------------------------
# Audio Extractor — Wav2Vec2-base
# ---------------------------------------------------------------------------

class AudioExtractor:
    """
    Extracts mean-pooled Wav2Vec2 embeddings from MELD video files.

    Audio is extracted from each .mp4 using FFmpeg (subprocess call).
    Converted to mono 16 kHz WAV before feeding to Wav2Vec2.

    Output shape per sample: [768]
    Missing/corrupt files → zero vector [768] with a warning.
    """

    def __init__(self, device: torch.device, batch_size: int = 16):
        logger.info(f"Loading audio encoder: {AUDIO_ENCODER_NAME}")
        self.processor = Wav2Vec2Processor.from_pretrained(AUDIO_ENCODER_NAME)
        self.model = Wav2Vec2Model.from_pretrained(AUDIO_ENCODER_NAME).to(device)
        _freeze_model(self.model)
        self.device = device
        self.batch_size = batch_size
        logger.info(f"  Audio encoder ready on {device}")

    def _extract_audio_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Use FFmpeg to extract audio from a video file.
        Returns a float32 numpy array at 16 kHz mono, or None on failure.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-ar", str(AUDIO_SAMPLE_RATE),
                    "-ac", "1",
                    "-f", "wav",
                    tmp_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            if result.returncode != 0:
                return None

            # Read WAV with soundfile (more robust than librosa for short clips)
            import soundfile as sf
            waveform, sr = sf.read(tmp_path, dtype="float32")
            if sr != AUDIO_SAMPLE_RATE:
                # Resample using librosa as fallback
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
            return waveform

        except Exception as e:
            logger.debug(f"Audio extraction failed for {video_path}: {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @torch.no_grad()
    def extract_single(self, video_path: Optional[str]) -> torch.Tensor:
        """
        Extract a [768] audio feature vector from one video file.
        Returns zeros if the video is missing, corrupt, or has no audio.
        """
        zero = torch.zeros(AUDIO_FEATURE_DIM, dtype=torch.float32)

        if video_path is None or not os.path.exists(video_path):
            return zero

        waveform = self._extract_audio_from_video(video_path)
        if waveform is None or len(waveform) < AUDIO_SAMPLE_RATE * 0.05:
            # Less than 50 ms of audio — treat as missing
            return zero

        try:
            inputs = self.processor(
                waveform,
                sampling_rate=AUDIO_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )

            input_values = inputs.input_values.to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            outputs = self.model(
                input_values,
                attention_mask=attention_mask,
            )

            # Attention-mask-aware mean pooling
            hidden = outputs.last_hidden_state  # [1, T, 768]
            if attention_mask is not None:
                # Expand mask to match hidden dim
                mask = attention_mask.unsqueeze(-1).float()
                # The hidden states have a different time dimension than input_values
                # Interpolate mask to match hidden state length
                mask_len = hidden.shape[1]
                mask = F.interpolate(
                    mask.permute(0, 2, 1),  # [1, 1, T_input]
                    size=mask_len,
                    mode="nearest",
                ).permute(0, 2, 1)  # [1, T_hidden, 1]
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden.mean(dim=1)  # [1, 768]

            return pooled.squeeze(0).cpu()  # [768]

        except Exception as e:
            logger.debug(f"Wav2Vec2 inference failed: {e}")
            return zero

    def extract_batch(
        self,
        video_paths: List[Optional[str]],
        desc: str = "Audio extraction",
    ) -> torch.Tensor:
        """
        Extract audio features for a list of video paths.

        Args:
            video_paths: List of .mp4 paths (or None for missing videos).
            desc:        tqdm progress bar description.

        Returns:
            FloatTensor [N, 768].
        """
        features = []
        missing_count = 0

        for path in tqdm(video_paths, desc=desc):
            feat = self.extract_single(path)
            if feat.sum().abs() < 1e-9:
                missing_count += 1
            features.append(feat)

        if missing_count > 0:
            logger.warning(
                f"  {missing_count}/{len(video_paths)} samples used zero audio features "
                "(video missing, no audio track, or extraction failed)."
            )

        return torch.stack(features, dim=0)  # [N, 768]


# ---------------------------------------------------------------------------
# Video Extractor — ViT-base-patch16-224
# ---------------------------------------------------------------------------

class VideoExtractor:
    """
    Extracts ViT [CLS] token embeddings from uniformly sampled video frames.

    Samples VIDEO_FRAMES_PER_CLIP (=8) frames from each .mp4 using OpenCV.
    Each frame is encoded by ViT-base. The 8 [CLS] embeddings are mean-pooled.

    Output shape per sample: [768]
    Missing/corrupt files → zero vector [768] with a warning.
    """

    def __init__(self, device: torch.device):
        logger.info(f"Loading video encoder: {VIDEO_ENCODER_NAME}")
        self.processor = ViTImageProcessor.from_pretrained(VIDEO_ENCODER_NAME)
        self.model = ViTModel.from_pretrained(VIDEO_ENCODER_NAME).to(device)
        _freeze_model(self.model)
        self.device = device
        self.n_frames = VIDEO_FRAMES_PER_CLIP
        logger.info(f"  Video encoder ready on {device}")

    def _sample_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        Uniformly sample VIDEO_FRAMES_PER_CLIP frames from a video.
        Returns a list of RGB numpy arrays (H, W, 3), or None on failure.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            return None

        # Uniform frame indices
        indices = np.linspace(0, max(total - 1, 0), num=self.n_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        if not frames:
            return None

        # If fewer frames were read than expected (short video), repeat last frame
        while len(frames) < self.n_frames:
            frames.append(frames[-1])

        return frames

    @torch.no_grad()
    def extract_single(self, video_path: Optional[str]) -> torch.Tensor:
        """
        Extract a [768] video feature vector from one video file.
        Returns zeros if the video is missing or corrupt.
        """
        zero = torch.zeros(VIDEO_FEATURE_DIM, dtype=torch.float32)

        if video_path is None or not os.path.exists(video_path):
            return zero

        try:
            frames = self._sample_frames(video_path)
            if frames is None:
                return zero

            # Process all frames together
            inputs = self.processor(images=frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)  # [N_frames, 3, 224, 224]

            outputs = self.model(pixel_values=pixel_values)

            # [CLS] token for each frame → mean pool
            cls_tokens = outputs.last_hidden_state[:, 0, :]  # [N_frames, 768]
            pooled = cls_tokens.mean(dim=0)  # [768]

            return pooled.cpu()

        except Exception as e:
            logger.debug(f"ViT inference failed for {video_path}: {e}")
            return zero

    def extract_batch(
        self,
        video_paths: List[Optional[str]],
        desc: str = "Video extraction",
    ) -> torch.Tensor:
        """
        Extract video features for a list of video paths.

        Args:
            video_paths: List of .mp4 paths (or None for missing videos).
            desc:        tqdm progress bar description.

        Returns:
            FloatTensor [N, 768].
        """
        features = []
        missing_count = 0

        for path in tqdm(video_paths, desc=desc):
            feat = self.extract_single(path)
            if feat.sum().abs() < 1e-9:
                missing_count += 1
            features.append(feat)

        if missing_count > 0:
            logger.warning(
                f"  {missing_count}/{len(video_paths)} samples used zero video features "
                "(video missing, corrupt frames, or ViT failure)."
            )

        return torch.stack(features, dim=0)  # [N, 768]


# ---------------------------------------------------------------------------
# Build Video Path List from DataFrame
# ---------------------------------------------------------------------------

def build_video_paths(
    df,  # pd.DataFrame from load_meld_metadata
    video_dir: Optional[str],
) -> List[Optional[str]]:
    """
    Build a list of video file paths for each row in a metadata DataFrame.

    Args:
        df:        Metadata DataFrame with 'sample_id' column.
        video_dir: Directory containing .mp4 files, or None.

    Returns:
        List of full file paths (or None if the file does not exist).
    """
    if video_dir is None or not os.path.isdir(video_dir):
        logger.warning(f"Video directory not available: {video_dir}")
        return [None] * len(df)

    paths = []
    for _, row in df.iterrows():
        fname = f"{row['sample_id']}.mp4"  # e.g. dia0_utt0.mp4
        full_path = os.path.join(video_dir, fname)
        paths.append(full_path if os.path.exists(full_path) else None)

    found = sum(1 for p in paths if p is not None)
    logger.info(f"Video paths: {found}/{len(paths)} found in {video_dir}")
    return paths
