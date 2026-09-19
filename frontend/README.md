# Affectra AI - Frontend

This is the React frontend for the Affectra AI multimodal emotion analysis system.

## Technology Stack
- **Framework**: React 18
- **Language**: TypeScript
- **Bundler**: Vite
- **Routing**: React Router v6
- **Styling**: Tailwind CSS v3
- **Icons**: Lucide React

## Folder Structure
```text
frontend/
├── public/           # Static assets
├── src/
│   ├── assets/       # Images, fonts
│   ├── components/
│   │   ├── layout/   # Sidebar, AppShell, BackendStatus
│   │   ├── ui/       # Reusable components (Button, Card)
│   │   └── common/   # EmptyState, ErrorState, LoadingState
│   ├── pages/        # Dashboard and feature placeholder pages
│   ├── services/     # API integration (Phase 6.2+)
│   ├── types/        # TypeScript interfaces (Phase 6.2+)
│   ├── lib/          # Utilities (cn for tailwind classes)
│   ├── App.tsx       # Routing
│   ├── main.tsx      # Entry point
│   └── index.css     # Global styles & Tailwind
├── .env.example      # Example environment variables
├── package.json      # Dependencies
├── tsconfig.json     # TypeScript configuration
└── vite.config.ts    # Vite configuration
```

## Installation
Ensure you have Node.js 22+ installed.
```bash
cd frontend
npm install
```

## Development
To start the Vite development server:
```bash
npm run dev
```

## Production Build
To create an optimized production build:
```bash
npm run build
```

## Environment Variables
Copy `.env.example` to `.env` and configure your local backend URL:
```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```
*Note: The frontend environment variables are not secret. Do not put LLM API keys or real secrets here.*

## Phase 6.2 Scope (Completed)
Phase 6.2 successfully connects the React application to the existing FastAPI backend using a reusable, strongly-typed API layer.

### Implemented:
- `BackendStatus` component dynamically pinging `/health`.
- `Knowledge` page fully wired to `/rag`, featuring retrieved context and generative answers.
- `Agent` page fully wired to `/agents`, capable of safety checking, trace execution, and generating final responses.
- Centralized, strongly-typed generic `apiClient` mapping to real Pydantic backend schemas.
- Minimal backend modification adding `CORSMiddleware` to `backend/app/main.py`.

### Architecture Note:
The API service architecture is split as follows:
- `src/types/api.ts`: TypeScript interfaces matching Python Pydantic schemas.
- `src/services/api/client.ts`: Reusable `fetch` client with consistent error mapping.
- `src/services/api/index.ts`: Endpoint-specific helper functions (`predict()`, `ragSearch()`, etc.).

## Phase 6.3 Scope (Completed)
Phase 6.3 successfully builds the Multimodal Analysis UI and connects raw frontend text/audio/video inputs to the real ML prediction pipeline.

### Implemented:
- **Multimodal Analysis Page (`Analysis.tsx`)**:
  - Modality switcher tabs for Text, Audio, Video, and Multimodal Fusion.
  - Text input with character counter and sample utterance presets.
  - Audio drag-and-drop file upload with specs validation and clear options.
  - Video drag-and-drop file upload with live video preview player and specs validation.
  - Interactive Analyze button with loading state ("Running neural feature extraction & fusion...").
  - Error alert state handling.
  - Neural ML Output Card rendering 100% real model predictions:
    - Predicted Emotion & Confidence percentage.
    - Full probability distribution bars for all 7 emotions (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`).
    - Predicted Sentiment & Confidence percentage.
    - Full probability distribution bars for all 3 sentiments (`positive`, `negative`, `neutral`).
- **Global Prediction Context (`PredictionContext.tsx`)**:
  - Automatically saves the latest prediction result and input summary to React state and `localStorage`.
  - Enables Phase 6.4 (Explainable AI, RAG, and Agents) to seamlessly consume the prediction output.
- **Backend Service Boundary (`feature_extraction_service.py` & `/predict/raw`)**:
  - Created `FeatureExtractionService` wrapping frozen pretrained encoders (`distilroberta-base`, `facebook/wav2vec2-base`, `google/vit-base-patch16-224`).
  - Added `POST /predict/raw` endpoint to process raw text strings, uploaded audio files, and uploaded video files into exact 768-d feature vectors before calling `model.pt`.
  - Kept production `models/affectra_multimodal/model.pt` completely locked and unchanged.

### Architecture Note:
- `src/types/api.ts`: TypeScript interfaces matching backend response models.
- `src/services/api/client.ts`: Extended with `postForm` for `multipart/form-data` uploads.
- `src/services/api/index.ts`: Added `predictRaw()` service method.
- `src/context/PredictionContext.tsx`: Shared state provider for prediction results.

**Next Phase**: Phase 6.4 — Explainable AI & Full Workflow Integration.

