"""
Affectra AI — Dataset Validation
==================================
Validates the MELD dataset before any training begins.

Checks performed:
  1.  CSV loading — all three splits can be read with pandas
  2.  Split size validation — expected row counts match documented values
  3.  Label validation — all emotion and sentiment labels are recognised
  4.  CSV column validation — required columns present
  5.  Video filename discovery — build a set of all found .mp4 files
  6.  CSV-to-video mapping — each row maps to expected video filename
  7.  Missing video detection — count rows with no corresponding video
  8.  Corrupt video detection — verify videos can be opened with cv2
  9.  Duplicate ID detection — flag duplicate (Dialogue_ID, Utterance_ID) pairs
  10. Label distribution — print class balance for emotion + sentiment

Outputs:
  dataset_validation_report.json — written to COLAB_OUTPUT_DIR
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd

from training.src.config import (
    COLAB_OUTPUT_DIR,
    EMOTION_LABELS,
    MELD_DEV_CSV,
    MELD_TEST_CSV,
    MELD_TRAIN_CSV,
    MELD_VIDEO_FILENAME_PATTERN,
    NUM_EMOTION_CLASSES,
    NUM_SENTIMENT_CLASSES,
    SENTIMENT_LABELS,
    normalise_emotion,
    normalise_sentiment,
)
from training.src.utils import get_logger, save_json

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Expected MELD CSV Columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Utterance",
    "Speaker",
    "Emotion",
    "Sentiment",
    "Dialogue_ID",
    "Utterance_ID",
]

# Approximate expected row counts per split (±10% tolerance)
EXPECTED_ROWS = {
    "train": 9989,
    "dev": 1109,
    "test": 2610,
}

# ---------------------------------------------------------------------------
# CSV Loading
# ---------------------------------------------------------------------------

def load_csv(csv_path: str, split: str) -> pd.DataFrame:
    """
    Load a MELD annotation CSV and verify required columns exist.

    Args:
        csv_path: Absolute path to the CSV file.
        split:    Split name ('train', 'dev', 'test') — for logging.

    Returns:
        pd.DataFrame with the loaded annotations.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError:        If required columns are missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    logger.info(f"Loaded {split} CSV: {len(df)} rows  ← {csv_path}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{split} CSV is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


# ---------------------------------------------------------------------------
# Individual Validation Checks
# ---------------------------------------------------------------------------

def validate_split_sizes(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Check that row counts are within 10% of the documented MELD values."""
    results = {}
    for split, df in dfs.items():
        actual = len(df)
        expected = EXPECTED_ROWS.get(split, 0)
        tolerance = int(expected * 0.10)
        ok = abs(actual - expected) <= tolerance
        results[split] = {
            "actual_rows": actual,
            "expected_rows": expected,
            "within_tolerance": ok,
            "status": "✅ OK" if ok else f"⚠️  WARNING — expected ~{expected}",
        }
        logger.info(
            f"  {split}: {actual} rows (expected ~{expected}) "
            f"→ {'OK' if ok else 'WARNING'}"
        )
    return results


def validate_labels(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Validate that every row's Emotion and Sentiment labels are recognised.
    Returns a report of unrecognised labels per split.
    """
    results = {}
    for split, df in dfs.items():
        bad_emotions: List[str] = []
        bad_sentiments: List[str] = []

        for _, row in df.iterrows():
            try:
                normalise_emotion(str(row["Emotion"]))
            except ValueError:
                bad_emotions.append(str(row["Emotion"]))
            try:
                normalise_sentiment(str(row["Sentiment"]))
            except ValueError:
                bad_sentiments.append(str(row["Sentiment"]))

        results[split] = {
            "bad_emotion_labels": list(set(bad_emotions)),
            "bad_sentiment_labels": list(set(bad_sentiments)),
            "bad_emotion_count": len(bad_emotions),
            "bad_sentiment_count": len(bad_sentiments),
            "status": (
                "✅ All labels valid"
                if not bad_emotions and not bad_sentiments
                else "⚠️  Unrecognised labels found"
            ),
        }

        if bad_emotions:
            logger.warning(
                f"  {split}: {len(bad_emotions)} unrecognised emotion labels: "
                f"{list(set(bad_emotions))}"
            )
        if bad_sentiments:
            logger.warning(
                f"  {split}: {len(bad_sentiments)} unrecognised sentiment labels: "
                f"{list(set(bad_sentiments))}"
            )
        if not bad_emotions and not bad_sentiments:
            logger.info(f"  {split}: All labels valid ✅")

    return results


def validate_duplicate_ids(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Check for duplicate (Dialogue_ID, Utterance_ID) pairs within each split."""
    results = {}
    for split, df in dfs.items():
        dupes = df.duplicated(subset=["Dialogue_ID", "Utterance_ID"])
        n_dupes = dupes.sum()
        results[split] = {
            "duplicate_count": int(n_dupes),
            "status": "✅ No duplicates" if n_dupes == 0 else f"⚠️  {n_dupes} duplicates",
        }
        if n_dupes > 0:
            logger.warning(f"  {split}: {n_dupes} duplicate (Dialogue_ID, Utterance_ID) pairs")
        else:
            logger.info(f"  {split}: No duplicate IDs ✅")
    return results


def get_video_filename(dialogue_id: int, utterance_id: int) -> str:
    """
    Build the canonical MELD video filename for a given dialogue/utterance.
    Pattern: dia{Dialogue_ID}_utt{Utterance_ID}.mp4
    """
    return MELD_VIDEO_FILENAME_PATTERN.format(
        dialogue_id=dialogue_id,
        utterance_id=utterance_id,
    )


def discover_video_files(video_dir: Optional[str]) -> set:
    """
    Recursively find all .mp4 files in a directory and return their
    base filenames as a set.

    Args:
        video_dir: Path to directory containing video files, or None.

    Returns:
        Set of .mp4 filenames (basename only, e.g. 'dia0_utt0.mp4').
    """
    if video_dir is None or not os.path.isdir(video_dir):
        logger.warning(f"Video directory not found: {video_dir}")
        return set()

    found = set()
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith(".mp4"):
                found.add(f)

    logger.info(f"  Discovered {len(found)} .mp4 files in {video_dir}")
    return found


def validate_video_mapping(
    df: pd.DataFrame,
    video_files: set,
    split: str,
    check_corrupt: bool = False,
    video_dir: Optional[str] = None,
    max_corrupt_checks: int = 50,
) -> Dict[str, Any]:
    """
    Verify the CSV-to-video mapping for a split.

    For each row in the CSV, construct the expected filename and check
    whether it exists in the discovered video file set.

    Args:
        df:                  Split DataFrame.
        video_files:         Set of discovered .mp4 filenames.
        split:               Split name (for logging).
        check_corrupt:       If True, try to open a sample of videos with cv2.
        video_dir:           Directory to look up full paths for corrupt checks.
        max_corrupt_checks:  Max videos to inspect for corruption.

    Returns:
        dict with mapping statistics.
    """
    missing_videos: List[str] = []
    found_count = 0
    corrupt_videos: List[str] = []

    checked_for_corruption = 0

    for _, row in df.iterrows():
        fname = get_video_filename(
            int(row["Dialogue_ID"]), int(row["Utterance_ID"])
        )

        if fname in video_files:
            found_count += 1

            # Optional: check a sample for corruption
            if (
                check_corrupt
                and video_dir
                and checked_for_corruption < max_corrupt_checks
            ):
                full_path = os.path.join(video_dir, fname)
                if os.path.exists(full_path):
                    cap = cv2.VideoCapture(full_path)
                    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) < 1:
                        corrupt_videos.append(fname)
                    cap.release()
                    checked_for_corruption += 1
        else:
            missing_videos.append(fname)

    total = len(df)
    missing_pct = (len(missing_videos) / total * 100) if total > 0 else 0

    result = {
        "total_rows": total,
        "found_videos": found_count,
        "missing_videos": len(missing_videos),
        "missing_pct": round(missing_pct, 2),
        "missing_filenames_sample": missing_videos[:20],  # first 20 for report
        "corrupt_videos_checked": checked_for_corruption,
        "corrupt_videos_found": len(corrupt_videos),
        "corrupt_filenames_sample": corrupt_videos[:10],
        "status": (
            "✅ All videos found"
            if not missing_videos
            else f"⚠️  {len(missing_videos)} ({missing_pct:.1f}%) missing"
        ),
    }

    if missing_videos:
        logger.warning(
            f"  {split}: {len(missing_videos)}/{total} videos missing "
            f"({missing_pct:.1f}%). These rows will use zero vectors."
        )
    else:
        logger.info(f"  {split}: All {total} videos found ✅")

    return result


def compute_label_distribution(
    dfs: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Compute emotion and sentiment class distribution for each split.
    Useful for computing weighted loss class weights.

    Returns:
        Nested dict: {split: {emotion: {label: count}, sentiment: {label: count}}}
    """
    distributions = {}
    for split, df in dfs.items():
        emotion_counts: Counter = Counter()
        sentiment_counts: Counter = Counter()

        for _, row in df.iterrows():
            try:
                emotion_counts[normalise_emotion(str(row["Emotion"]))] += 1
            except ValueError:
                emotion_counts["UNKNOWN"] += 1
            try:
                sentiment_counts[normalise_sentiment(str(row["Sentiment"]))] += 1
            except ValueError:
                sentiment_counts["UNKNOWN"] += 1

        distributions[split] = {
            "emotion": dict(emotion_counts),
            "sentiment": dict(sentiment_counts),
            "total": len(df),
        }

        logger.info(f"\n  {split.upper()} label distribution:")
        logger.info(f"    Emotions:   {dict(emotion_counts)}")
        logger.info(f"    Sentiments: {dict(sentiment_counts)}")

    return distributions


# ---------------------------------------------------------------------------
# Main Validation Runner
# ---------------------------------------------------------------------------

def validate_dataset(
    meld_root: str,
    video_dirs: Optional[Dict[str, Optional[str]]] = None,
    check_corrupt_videos: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run all validation checks and produce a JSON report.

    Args:
        meld_root:            Directory containing annotation CSVs.
        video_dirs:           Dict mapping split names to video directories
                              (or None if videos not found).
        check_corrupt_videos: Inspect a sample of videos for corruption.
        output_dir:           Where to save the JSON report.

    Returns:
        Full validation report dictionary.
    """
    logger.info("=" * 60)
    logger.info("AFFECTRA AI — MELD DATASET VALIDATION")
    logger.info("=" * 60)

    report: Dict[str, Any] = {
        "meld_root": meld_root,
        "checks": {},
    }

    # ── 1. Load CSVs ───────────────────────────────────────────────────────
    logger.info("\n[1/10] Loading annotation CSVs...")
    try:
        dfs = {
            "train": load_csv(os.path.join(meld_root, MELD_TRAIN_CSV), "train"),
            "dev":   load_csv(os.path.join(meld_root, MELD_DEV_CSV), "dev"),
            "test":  load_csv(os.path.join(meld_root, MELD_TEST_CSV), "test"),
        }
        report["checks"]["csv_loading"] = "✅ All 3 CSVs loaded"
    except (FileNotFoundError, ValueError) as e:
        report["checks"]["csv_loading"] = f"❌ FAILED: {e}"
        logger.error(f"CSV loading failed: {e}")
        report["overall_status"] = "❌ FAILED — cannot proceed"
        _save_report(report, output_dir)
        return report

    # ── 2. Split size validation ───────────────────────────────────────────
    logger.info("\n[2/10] Validating split sizes...")
    report["checks"]["split_sizes"] = validate_split_sizes(dfs)

    # ── 3. Label validation ────────────────────────────────────────────────
    logger.info("\n[3/10] Validating labels...")
    report["checks"]["label_validation"] = validate_labels(dfs)

    # ── 4. Duplicate ID detection ──────────────────────────────────────────
    logger.info("\n[4/10] Checking for duplicate IDs...")
    report["checks"]["duplicate_ids"] = validate_duplicate_ids(dfs)

    # ── 5. Label distribution ──────────────────────────────────────────────
    logger.info("\n[5/10] Computing label distribution...")
    report["label_distribution"] = compute_label_distribution(dfs)

    # ── 6–8. Video mapping (per split) ────────────────────────────────────
    if video_dirs is None:
        video_dirs = {"train": None, "dev": None, "test": None}

    logger.info("\n[6/10] Discovering video files...")
    video_file_sets = {
        split: discover_video_files(video_dirs.get(split))
        for split in ["train", "dev", "test"]
    }

    logger.info("\n[7/10] Checking CSV-to-video mapping...")
    report["checks"]["video_mapping"] = {}
    for split, df in dfs.items():
        report["checks"]["video_mapping"][split] = validate_video_mapping(
            df,
            video_files=video_file_sets[split],
            split=split,
            check_corrupt=check_corrupt_videos,
            video_dir=video_dirs.get(split),
        )

    # ── 9. Corrupt video detection ─────────────────────────────────────────
    if check_corrupt_videos:
        logger.info("\n[8/10] Corrupt video detection (already run above).")
    else:
        logger.info("\n[8/10] Corrupt video check: SKIPPED (set check_corrupt_videos=True to enable).")

    # ── 10. Summary ────────────────────────────────────────────────────────
    logger.info("\n[9/10] Computing overall status...")
    any_error = any(
        "❌" in str(v) for v in report["checks"].values()
    )
    report["overall_status"] = (
        "❌ FAILED — see checks above"
        if any_error
        else "✅ Dataset is ready for training"
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"VALIDATION COMPLETE: {report['overall_status']}")
    logger.info("=" * 60)

    # ── Save report ────────────────────────────────────────────────────────
    _save_report(report, output_dir)
    return report


def _save_report(report: Dict[str, Any], output_dir: Optional[str]) -> None:
    """Internal helper to save the validation report to JSON."""
    save_dir = output_dir or COLAB_OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, "dataset_validation_report.json")
    save_json(report, report_path)
    logger.info(f"📋 Validation report saved: {report_path}")
