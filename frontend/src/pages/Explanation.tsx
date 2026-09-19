import React, { useState } from 'react'
import { Brain, Sparkles, Server } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { apiService } from '@/services/api'
import { usePrediction } from '@/context/PredictionContext'
import type { ExplanationResponse } from '@/types/api'

export const Explanation: React.FC = () => {
  const { prediction } = usePrediction()
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ExplanationResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleExplain = async () => {
    if (!prediction) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await apiService.explain({
        emotion_label: prediction.emotion.label,
        emotion_probabilities: prediction.emotion.probabilities,
        sentiment_label: prediction.sentiment.label,
        sentiment_probabilities: prediction.sentiment.probabilities,
      })
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred while generating explanation.')
    } finally {
      setLoading(false)
    }
  }

  if (!prediction) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center animate-in fade-in duration-500">
        <div className="w-24 h-24 rounded-3xl flex items-center justify-center mb-8 shadow-sm bg-secondary-light">
          <Brain className="w-12 h-12 text-secondary" />
        </div>
        <h1 className="text-3xl font-bold text-text-main mb-4">Explainable AI</h1>
        <p className="text-lg text-text-muted max-w-lg mx-auto mb-8">
          Run an analysis first.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto animate-in fade-in duration-500">
      <div className="flex items-center gap-4 mb-8">
        <div className="w-12 h-12 rounded-2xl bg-secondary-light flex items-center justify-center shadow-sm">
          <Brain className="w-6 h-6 text-secondary" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-text-main">Explainable AI</h1>
          <p className="text-text-muted mt-1">Generate a natural language explanation for the latest prediction.</p>
        </div>
      </div>

      <Card className="shadow-sm border-secondary/20">
        <CardHeader className="bg-secondary-light/30 pb-4">
          <CardTitle className="text-xl flex items-center gap-2">
            Prediction Context
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="p-4 rounded-lg bg-background/50 border border-border">
              <div className="text-sm text-text-muted mb-1">Emotion</div>
              <div className="font-semibold text-text-main capitalize">{prediction.emotion.label}</div>
            </div>
            <div className="p-4 rounded-lg bg-background/50 border border-border">
              <div className="text-sm text-text-muted mb-1">Sentiment</div>
              <div className="font-semibold text-text-main capitalize">{prediction.sentiment.label}</div>
            </div>
          </div>
          <Button
            onClick={handleExplain}
            className="w-full h-12 bg-secondary hover:bg-secondary/90 text-white focus:ring-2 focus:ring-secondary/50 focus:outline-none transition-all"
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                Generating Explanation...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <Sparkles className="w-4 h-4" />
                Generate Explanation
              </span>
            )}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <div className="p-4 rounded-lg bg-error-light text-error text-sm border border-error/20 flex items-center gap-2">
          <Server className="w-4 h-4" />
          {error}
        </div>
      )}

      {result && (
        <Card className="mt-8 overflow-hidden border-secondary/20 animate-in slide-in-from-bottom-4 duration-500">
          <div className="h-1 w-full bg-gradient-to-r from-secondary to-purple-400"></div>
          <CardHeader className="bg-secondary-light/30 pb-4">
            <CardTitle className="text-xl flex items-center gap-2">
              <Brain className="w-5 h-5 text-secondary" /> LLM Explanation
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <p className="text-text-main leading-relaxed whitespace-pre-wrap">{result.explanation}</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
