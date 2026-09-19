export interface EmotionResult {
  label: string;
  probabilities: Record<string, number>;
}

export interface SentimentResult {
  label: string;
  probabilities: Record<string, number>;
}

export interface PredictionRequest {
  text_feat: number[]; // length 768
  audio_feat: number[]; // length 768
  video_feat: number[]; // length 768
}

export interface PredictionResponse {
  emotion: EmotionResult;
  sentiment: SentimentResult;
}

export interface ExplanationRequest {
  emotion_label: string;
  emotion_probabilities: Record<string, number>;
  sentiment_label: string;
  sentiment_probabilities: Record<string, number>;
}

export interface ExplanationResponse {
  explanation: string;
}

export interface RAGRequest {
  question: string;
  top_k?: number; // defaults to 3
}

export interface RAGContextItem {
  source: string;
  chunk_index: number;
  score: number;
  text: string;
}

export interface RAGResponse {
  question: string;
  answer: string;
  retrieved_context: RAGContextItem[];
}

export interface AgentAPIRequest {
  user_query?: string | null;
  emotion_label?: string | null;
  emotion_probabilities?: Record<string, number> | null;
  sentiment_label?: string | null;
  sentiment_probabilities?: Record<string, number> | null;
  top_k?: number; // defaults to 3
}

export interface AgentContextItem {
  source: string;
  chunk_index: number;
  score: number;
  text: string;
}

export interface AgentAPIResponse {
  final_response: string;
  safety_approved: boolean;
  execution_trace: string[];
  retrieved_context: AgentContextItem[];
  error_message?: string | null;
}

export interface HealthResponse {
  status: string;
  service: string;
  model_loaded: boolean;
}
