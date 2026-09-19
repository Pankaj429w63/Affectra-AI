import React, { createContext, useContext, useState, type ReactNode } from 'react';
import type { PredictionResponse } from '@/types/api';

export interface InputSummary {
  text?: string;
  audioName?: string;
  videoName?: string;
  modalitiesUsed: ('text' | 'audio' | 'video')[];
  timestamp: string;
}

interface PredictionContextType {
  prediction: PredictionResponse | null;
  inputSummary: InputSummary | null;
  setPredictionResult: (result: PredictionResponse, summary: InputSummary) => void;
  clearPrediction: () => void;
}

const PredictionContext = createContext<PredictionContextType | undefined>(undefined);

export const PredictionProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(() => {
    const saved = localStorage.getItem('affectra_latest_prediction');
    return saved ? JSON.parse(saved) : null;
  });

  const [inputSummary, setInputSummary] = useState<InputSummary | null>(() => {
    const saved = localStorage.getItem('affectra_latest_input_summary');
    return saved ? JSON.parse(saved) : null;
  });

  const setPredictionResult = (result: PredictionResponse, summary: InputSummary) => {
    setPrediction(result);
    setInputSummary(summary);
    localStorage.setItem('affectra_latest_prediction', JSON.stringify(result));
    localStorage.setItem('affectra_latest_input_summary', JSON.stringify(summary));
  };

  const clearPrediction = () => {
    setPrediction(null);
    setInputSummary(null);
    localStorage.removeItem('affectra_latest_prediction');
    localStorage.removeItem('affectra_latest_input_summary');
  };

  return (
    <PredictionContext.Provider
      value={{
        prediction,
        inputSummary,
        setPredictionResult,
        clearPrediction,
      }}
    >
      {children}
    </PredictionContext.Provider>
  );
};

export const usePrediction = () => {
  const context = useContext(PredictionContext);
  if (!context) {
    throw new Error('usePrediction must be used within a PredictionProvider');
  }
  return context;
};
