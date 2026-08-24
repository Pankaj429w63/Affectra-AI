"""
Affectra AI — MELD Dataset Download
=====================================
Handles downloading the raw MELD archive and cloning the official
annotation repository directly inside Google Colab.

IMPORTANT:
  - These functions must run inside Google Colab, not on your local machine.
  - The MELD dataset (~11 GB) is downloaded to /content/meld_data/
  - Annotation CSVs are cloned from the official GitHub repository.
  - Nothing is stored permanently without Google Drive mounted first.

Usage (inside Colab notebook cell):
    from training.src.download import download_meld, find_meld_root
    download_meld()
    meld_root = find_meld_root()
"""

import os
import subprocess
import tarfile
from pathlib import Path
from typing import Optional

from training.src.config import (
    COLAB_DATA_DIR,
    MELD_ANNOTATION_REPO,
    MELD_DEV_CSV,
    MELD_RAW_URL,
    MELD_TEST_CSV,
    MELD_TRAIN_CSV,
)
from training.src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Download Constants
# ---------------------------------------------------------------------------

MELD_ARCHIVE_PATH = "/content/MELD.Raw.tar.gz"
MELD_ANNOTATION_DIR = "/content/MELD_annotations"

# Required annotation CSV filenames
REQUIRED_CSVS = [MELD_TRAIN_CSV, MELD_DEV_CSV, MELD_TEST_CSV]


# ---------------------------------------------------------------------------
# Download Functions
# ---------------------------------------------------------------------------

def download_meld_archive(force: bool = False) -> str:
    """
    Download the raw MELD .tar.gz archive into Colab's local storage.
    Skips the download if the archive already exists (unless force=True).

    Args:
        force: Re-download even if the archive already exists.

    Returns:
        str: Path to the downloaded archive file.

    Raises:
        RuntimeError: If the download fails.
    """
    if os.path.exists(MELD_ARCHIVE_PATH) and not force:
        size_gb = os.path.getsize(MELD_ARCHIVE_PATH) / (1024 ** 3)
        logger.info(
            f"Archive already exists ({size_gb:.2f} GB): {MELD_ARCHIVE_PATH}"
        )
        return MELD_ARCHIVE_PATH

    logger.info(f"Downloading MELD archive from: {MELD_RAW_URL}")
    logger.info("This is ~11 GB — please be patient (may take 10–20 minutes).")

    result = subprocess.run(
        ["wget", "-q", "--show-progress", "-O", MELD_ARCHIVE_PATH, MELD_RAW_URL],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"wget failed with return code {result.returncode}. "
            "Check your internet connection and the download URL."
        )

    size_gb = os.path.getsize(MELD_ARCHIVE_PATH) / (1024 ** 3)
    logger.info(f"Download complete: {MELD_ARCHIVE_PATH} ({size_gb:.2f} GB)")
    return MELD_ARCHIVE_PATH


def extract_meld_archive(archive_path: str, force: bool = False) -> str:
    """
    Extract the MELD archive into COLAB_DATA_DIR.
    Inspects the actual extracted directory structure automatically
    (does not assume a hard-coded sub-folder name).

    Args:
        archive_path: Path to the .tar.gz archive.
        force:        Re-extract even if the directory already exists.

    Returns:
        str: Path to the root extracted directory.

    Raises:
        FileNotFoundError: If the archive does not exist.
        RuntimeError:      If extraction fails.
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    os.makedirs(COLAB_DATA_DIR, exist_ok=True)

    # Check if extraction already done
    existing = list(Path(COLAB_DATA_DIR).iterdir())
    if existing and not force:
        logger.info(
            f"Extraction directory already contains {len(existing)} items — "
            "skipping extraction."
        )
        return COLAB_DATA_DIR

    logger.info(f"Extracting {archive_path} → {COLAB_DATA_DIR}")
    logger.info("This may take several minutes...")

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=COLAB_DATA_DIR)
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}") from e

    logger.info(f"Extraction complete: {COLAB_DATA_DIR}")
    return COLAB_DATA_DIR


def inspect_extracted_structure(data_dir: str) -> None:
    """
    Print the full directory tree of the extracted MELD data directory
    (up to 3 levels deep). This helps confirm the actual structure
    rather than relying on assumptions.

    Args:
        data_dir: Root directory to inspect.
    """
    logger.info(f"=== Inspecting MELD directory structure: {data_dir} ===")
    for root, dirs, files in os.walk(data_dir):
        # Limit depth to 3 levels for readability
        depth = root.replace(data_dir, "").count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        indent = "  " * depth
        logger.info(f"{indent}{os.path.basename(root)}/")
        if depth < 3:
            for f in files[:5]:  # Show up to 5 files per directory
                logger.info(f"{indent}  {f}")
            if len(files) > 5:
                logger.info(f"{indent}  ... ({len(files) - 5} more files)")


def find_meld_root(data_dir: str = COLAB_DATA_DIR) -> str:
    """
    Automatically locate the MELD root directory by searching for
    the required CSV files. Does NOT assume a fixed subdirectory name.

    Args:
        data_dir: The directory where MELD was extracted.

    Returns:
        str: Path to the directory containing the annotation CSVs.

    Raises:
        FileNotFoundError: If the CSVs cannot be found anywhere under data_dir.
    """
    logger.info(f"Searching for MELD annotation CSVs under: {data_dir}")

    for root, dirs, files in os.walk(data_dir):
        if MELD_TRAIN_CSV in files:
            logger.info(f"Found MELD root: {root}")
            # Verify all required CSVs are present
            missing = [csv for csv in REQUIRED_CSVS if csv not in files]
            if missing:
                logger.warning(
                    f"Found {MELD_TRAIN_CSV} at {root}, but missing: {missing}"
                )
            return root

    raise FileNotFoundError(
        f"Could not find '{MELD_TRAIN_CSV}' anywhere under '{data_dir}'. "
        "Did the download and extraction succeed? "
        "Try running inspect_extracted_structure() to see what was extracted."
    )


def find_video_dir(meld_root: str, split: str) -> Optional[str]:
    """
    Locate the directory containing video files for a specific split.
    MELD video files follow the pattern: dia{D}_utt{U}.mp4

    The actual subdirectory name may vary between MELD releases.
    This function searches for the directory containing .mp4 files
    matching the split name.

    Args:
        meld_root: Directory containing annotation CSVs.
        split:     One of 'train', 'dev', 'test'.

    Returns:
        str or None: Path to the video directory, or None if not found.
    """
    # Common MELD directory naming patterns for videos
    candidate_names = [
        split,
        f"{split}_splits",
        f"MELD.Raw/{split}",
        f"output_repeated_splits_{split}",
        f"{split}_converted",
    ]

    # First: check common relative paths under meld_root
    for name in candidate_names:
        candidate = os.path.join(meld_root, name)
        if os.path.isdir(candidate) and any(
            f.endswith(".mp4") for f in os.listdir(candidate)
        ):
            logger.info(f"Found video dir for '{split}': {candidate}")
            return candidate

    # Second: walk up and search more broadly
    search_root = os.path.dirname(meld_root)
    for root, dirs, files in os.walk(search_root):
        mp4_files = [f for f in files if f.endswith(".mp4")]
        if len(mp4_files) > 10 and split in root.lower():
            logger.info(f"Found video dir for '{split}': {root}")
            return root

    logger.warning(
        f"Could not locate video directory for split='{split}' under {meld_root}. "
        "Audio/video features will use zero vectors for all samples in this split."
    )
    return None


def clone_annotation_repo(force: bool = False) -> str:
    """
    Clone the official MELD annotation repository to get the authoritative
    CSV files. This is a fallback if the archive CSVs are missing or corrupt.

    Args:
        force: Re-clone even if the directory already exists.

    Returns:
        str: Path to the cloned annotation repository.
    """
    if os.path.exists(MELD_ANNOTATION_DIR) and not force:
        logger.info(f"Annotation repo already exists: {MELD_ANNOTATION_DIR}")
        return MELD_ANNOTATION_DIR

    logger.info(f"Cloning official MELD annotations from: {MELD_ANNOTATION_REPO}")
    result = subprocess.run(
        [
            "git", "clone", "--depth", "1",
            MELD_ANNOTATION_REPO,
            MELD_ANNOTATION_DIR,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed: {result.stderr}. "
            "Check your internet connection."
        )

    logger.info(f"Annotations cloned: {MELD_ANNOTATION_DIR}")
    return MELD_ANNOTATION_DIR


def download_meld(force_download: bool = False, force_extract: bool = False) -> dict:
    """
    Complete MELD download pipeline:
      1. Download the raw archive
      2. Extract into Colab storage
      3. Inspect the extracted structure
      4. Find the MELD root directory
      5. Return paths for train/dev/test

    Args:
        force_download: Re-download even if archive exists.
        force_extract:  Re-extract even if directory exists.

    Returns:
        dict with keys: 'meld_root', 'train_csv', 'dev_csv', 'test_csv',
                        'train_video_dir', 'dev_video_dir', 'test_video_dir'
    """
    # Step 1: Download
    archive = download_meld_archive(force=force_download)

    # Step 2: Extract
    extract_meld_archive(archive, force=force_extract)

    # Step 3: Inspect
    inspect_extracted_structure(COLAB_DATA_DIR)

    # Step 4: Find root
    meld_root = find_meld_root(COLAB_DATA_DIR)

    # Step 5: Build path dictionary
    paths = {
        "meld_root": meld_root,
        "train_csv": os.path.join(meld_root, MELD_TRAIN_CSV),
        "dev_csv": os.path.join(meld_root, MELD_DEV_CSV),
        "test_csv": os.path.join(meld_root, MELD_TEST_CSV),
        "train_video_dir": find_video_dir(meld_root, "train"),
        "dev_video_dir": find_video_dir(meld_root, "dev"),
        "test_video_dir": find_video_dir(meld_root, "test"),
    }

    logger.info("=== MELD paths resolved ===")
    for k, v in paths.items():
        logger.info(f"  {k}: {v}")

    return paths
