import React, { useState } from 'react'
import { MessageSquare, Send, Server, Shield, ShieldAlert, Cpu, BookOpen } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { apiService } from '@/services/api'
import { usePrediction } from '@/context/PredictionContext'
import type { AgentAPIResponse, AgentAPIRequest } from '@/types/api'
import { cn } from '@/lib/utils'

export const Agent: React.FC = () => {
  const { prediction } = usePrediction()
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AgentAPIResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleChat = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const requestData: AgentAPIRequest = {
        user_query: query,
        top_k: 3,
      }
      
      if (prediction) {
        requestData.emotion_label = prediction.emotion.label
        requestData.emotion_probabilities = prediction.emotion.probabilities
        requestData.sentiment_label = prediction.sentiment.label
        requestData.sentiment_probabilities = prediction.sentiment.probabilities
      }

      const response = await apiService.chatAgent(requestData)
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6 max-w-4xl mx-auto animate-in fade-in duration-500">
      <div className="flex items-center gap-4 mb-8">
        <div className="w-12 h-12 rounded-2xl bg-warning-light flex items-center justify-center shadow-sm">
          <MessageSquare className="w-6 h-6 text-warning" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-text-main">AI Agent</h1>
          <p className="text-text-muted mt-1">Chat with our intelligent, safe AI agent pipeline.</p>
        </div>
      </div>

      <Card className="shadow-sm">
        <CardContent className="pt-6">
          <form onSubmit={handleChat} className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask anything..."
                aria-label="Ask anything"
                className="w-full h-12 px-4 rounded-lg border border-border focus:border-warning focus:ring-1 focus:ring-warning outline-none transition-all"
                disabled={loading}
              />
            </div>
            <Button 
              type="submit" 
              className="w-full sm:w-auto h-12 px-8 bg-warning hover:bg-warning/90 text-white focus:ring-2 focus:ring-warning/50 focus:outline-none"
              disabled={!query.trim() || loading}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  Processing...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <Send className="w-4 h-4" />
                  Chat
                </span>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <div className="p-4 rounded-lg bg-error-light text-error text-sm border border-error/20 flex items-center gap-2">
          <Server className="w-4 h-4" />
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6 mt-8 animate-in slide-in-from-bottom-4 duration-500">
          <Card className={cn(
            "overflow-hidden transition-colors border",
            result.safety_approved ? "border-warning/20" : "border-error/50"
          )}>
            <div className={cn(
              "h-1 w-full bg-gradient-to-r",
              result.safety_approved ? "from-warning to-yellow-300" : "from-error to-red-400"
            )}></div>
            <CardHeader className="bg-background/50 pb-4 flex flex-row items-center justify-between">
              <CardTitle className="text-xl flex items-center gap-2">
                <MessageSquare className={cn("w-5 h-5", result.safety_approved ? "text-warning" : "text-error")} /> 
                Agent Response
              </CardTitle>
              {result.safety_approved ? (
                <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-success-light text-success text-xs font-semibold">
                  <Shield className="w-3.5 h-3.5" /> Safety Approved
                </div>
              ) : (
                <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-error-light text-error text-xs font-semibold">
                  <ShieldAlert className="w-3.5 h-3.5" /> Safety Blocked
                </div>
              )}
            </CardHeader>
            <CardContent className="pt-6">
              <p className="text-text-main leading-relaxed whitespace-pre-wrap">{result.final_response}</p>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-background/50">
              <CardHeader className="py-4 border-b border-border">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-text-muted" /> Execution Trace
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="space-y-3">
                  {result.execution_trace.map((step, idx) => (
                    <div key={idx} className="flex items-center gap-3">
                      <div className="w-6 h-6 rounded-full bg-white border border-border flex items-center justify-center text-xs font-medium text-text-muted">
                        {idx + 1}
                      </div>
                      <span className="text-sm font-medium text-text-main">{step}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-background/50">
              <CardHeader className="py-4 border-b border-border">
                <CardTitle className="text-sm flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-text-muted" /> Knowledge Context
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4">
                {result.retrieved_context.length > 0 ? (
                  <div className="space-y-3">
                    {result.retrieved_context.map((ctx, idx) => (
                      <div key={idx} className="text-sm p-3 bg-white border border-border rounded-lg">
                        <div className="font-medium text-text-main mb-1 truncate">{ctx.source}</div>
                        <p className="text-text-muted line-clamp-2 text-xs">{ctx.text}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-text-muted italic">No external context retrieved for this query.</p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}
