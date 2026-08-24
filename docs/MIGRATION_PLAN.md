# Affectra AI — Repository Migration Plan

> **Date:** 2026-08-24  
> **Author:** Pankaj Yadav  
> **Goal:** Rebuild the repository from a single-file Emotion Detection prototype into a clean, scalable **Multimodal Emotion Intelligence Platform** — while preserving commit history and the existing GitHub remote.

---

## 1. Current Repository State (Pre-Migration)

### 1.1 Root-Level Files

| File | Size | Description |
|---|---|---|
| `app.py` | 6.3 KB | Flask REST API with DummyModel for simulated inference |
| `config.py` | 1.2 KB | Global config: emotion/sentiment labels, feature dims, hyperparams |
| `gradio_demo.py` | 4.1 KB | Gradio UI demo that wraps the model for browser interaction |
| `index.html` | 38 KB | Standalone HTML frontend page (not connected to any bundler) |
| `main.py` | 1.7 KB | CLI entry point: delegates to `models/src/train.py` |
| `predict.py` | 4.3 KB | CLI prediction script using dummy audio/video features |
| `dia446_utt7.mp4` | 729 KB | Raw video sample from the MELD dataset — committed to Git |
| `dia945_utt12.mp4` | 412 KB | Raw video sample from the MELD dataset — committed to Git |
| `requirements.txt` | 68 B | Minimal Python dependency list |
| `.env` | 0 B | Empty env file — tracked by Git (security risk) |
| `.gitignore` | 4.9 KB | Generic Python `.gitignore`; lacks ML/data-specific rules |
| `README.md` | 7.4 KB | Documents the old "Emotion Detection" project |
| `LICENSE` | 1.1 KB | MIT License — **KEEP** |

### 1.2 Directories

| Directory | Description |
|---|---|
| `.git/` | Git metadata — **KEEP** |
| `python/` | Empty placeholder directory — no files inside |

### 1.3 Git History

```
5cf3856  Revise README for Emotion Detection project
34f2b7b  Add files via upload
c65bc06  first commit
```

### 1.4 Remote

```
origin  https://github.com/Pankaj429w63/Affectra-AI.git (fetch/push)
```

---

## 2. Files to Remove and Why They Are Obsolete

| File / Dir | Reason for Removal |
|---|---|
| `app.py` | Prototype Flask API using `DummyModel` — will be rebuilt properly inside `backend/` |
| `config.py` | Flat config file; replaced by structured per-module configs in `backend/` and `training/` |
| `gradio_demo.py` | Quick-demo Gradio script; superseded by the proper `frontend/` application |
| `index.html` | 38 KB standalone HTML with no build pipeline — will be rebuilt inside `frontend/` |
| `main.py` | Monolithic CLI entry point; training pipeline moves to `training/src/` |
| `predict.py` | Duplicates inference logic from `app.py`; inference will be a backend service |
| `dia446_utt7.mp4` | Raw MELD dataset video clip committed to Git — violates dataset licensing and inflates repo size |
| `dia945_utt12.mp4` | Same as above |
| `requirements.txt` | Will be replaced per-component (`backend/requirements.txt`, `training/requirements.txt`) |
| `python/` | Empty directory with no content |
| `.env` | Must be removed from Git tracking; secrets must never be committed |

---

## 3. Security: `.env` Handling

- `.env` is currently **tracked by Git** (present in `git ls-files`).
- Action: run `git rm --cached .env` to untrack without deleting the local file.
- Ensure `.env` is listed in `.gitignore` (it already is, but the file was committed before the rule was respected).
- Create `.env.example` with **safe placeholder values only** — no real secrets.

---

## 4. New Project Structure

```
Affectra-AI/
├── backend/                    # FastAPI / Flask inference service
│   └── (to be implemented)
├── frontend/                   # React / Next.js or HTML/CSS/JS UI
│   └── (to be implemented)
├── training/                   # Model training pipeline
│   ├── src/                    # Training source code (data loaders, model, trainer)
│   └── notebooks/              # Jupyter notebooks for experimentation & EDA
├── models/                     # Saved model weights & checkpoints (git-ignored)
│   └── .gitkeep
├── data/                       # Datasets and raw files (git-ignored)
│   └── .gitkeep
├── docs/                       # Project documentation
│   └── MIGRATION_PLAN.md       # This file
├── scripts/                    # Utility scripts (setup, download, export)
├── .env.example                # Safe placeholder env file
├── .gitignore                  # Updated for ML project (datasets, checkpoints, venvs)
├── README.md                   # Updated project description
└── LICENSE                     # MIT License (unchanged)
```

### 4.1 Directory Purpose Summary

| Directory | Purpose |
|---|---|
| `backend/` | REST API server for real-time emotion inference (multimodal input -> prediction) |
| `frontend/` | User-facing web application for interacting with the Affectra AI platform |
| `training/src/` | Clean, modular training code: data loaders, model architectures, trainers, evaluators |
| `training/notebooks/` | EDA, quick experiments, visualisation notebooks |
| `models/` | Stores `.pt` / `.onnx` model checkpoints — **git-ignored**, managed separately |
| `data/` | Raw and processed datasets (MELD, etc.) — **git-ignored**, never committed |
| `docs/` | Architecture docs, design decisions, migration notes |
| `scripts/` | One-off automation: dataset download, environment setup, model export |

---

## 5. .gitignore Rules Added

New rules appended for the ML platform context:

```
# Secrets
.env

# Datasets & raw data
data/
datasets/
*.mp4
*.avi
*.mov

# Model weights & checkpoints
models/*.pt
models/*.onnx
models/*.bin
checkpoints/
*.ckpt

# Training outputs
runs/
outputs/
lightning_logs/
wandb/

# Python environments
venv/
.venv/
env/
__pycache__/
*.pyc

# Node / frontend
node_modules/
.next/
dist/
build/

# Logs & caches
*.log
logs/
.cache/
```

---

## 6. Migration Execution Steps

- [x] Inspect current repository
- [x] Create `docs/MIGRATION_PLAN.md`
- [ ] Remove obsolete files (`app.py`, `config.py`, `gradio_demo.py`, `index.html`, `main.py`, `predict.py`, `requirements.txt`, `python/`, `*.mp4`)
- [ ] Untrack `.env` from Git (`git rm --cached .env`)
- [ ] Create new directory structure
- [ ] Write updated `.gitignore`
- [ ] Write `README.md` with new project title and description
- [ ] Write `.env.example` with placeholder values
- [ ] Commit with message: `chore: rebuild repository structure for Affectra AI`
- [ ] Push to `origin/main`

---

## 7. What Is Preserved

| Item | Reason |
|---|---|
| `.git/` | Full commit history and remote connection |
| `LICENSE` | MIT License remains valid for the rebuilt project |
| `origin` remote | `https://github.com/Pankaj429w63/Affectra-AI.git` — push target unchanged |

---

*This migration does not implement any application code. It only establishes a clean, production-ready repository scaffold for Affectra AI.*
