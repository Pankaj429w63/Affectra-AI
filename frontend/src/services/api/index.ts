import { apiClient } from './client';
import type {
  HealthResponse,
  PredictionRequest,
  PredictionResponse,
  ExplanationRequest,
  ExplanationResponse,
  RAGRequest,
  RAGResponse,
  AgentAPIRequest,
  AgentAPIResponse
} from '@/types/api';

export const apiService = {
  checkHealth: () => {
    return apiClient.get<HealthResponse>('/health');
  },

  predict: (data: PredictionRequest) => {
    return apiClient.post<PredictionResponse>('/predict', data);
  },

  predictRaw: (formData: FormData) => {
    return apiClient.postForm<PredictionResponse>('/predict/raw', formData);
  },

  explain: (data: ExplanationRequest) => {
    return apiClient.post<ExplanationResponse>('/explain', data);
  },

  ragSearch: (data: RAGRequest) => {
    return apiClient.post<RAGResponse>('/rag', data);
  },

  chatAgent: (data: AgentAPIRequest) => {
    return apiClient.post<AgentAPIResponse>('/agents', data);
  }
};
