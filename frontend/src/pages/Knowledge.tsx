import React, { useState } from 'react'
import { BookOpen, Search, Server } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { apiService } from '@/services/api'
import type { RAGResponse } from '@/types/api'

export const Knowledge: React.FC = () => {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<RAGResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await apiService.ragSearch({ question: query, top_k: 3 })
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
        <div className="w-12 h-12 rounded-2xl bg-success-light flex items-center justify-center shadow-sm">
          <BookOpen className="w-6 h-6 text-success" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-text-main">Knowledge Base (RAG)</h1>
          <p className="text-text-muted mt-1">Ask questions about Affectra AI using retrieved context.</p>
        </div>
      </div>

      <Card className="shadow-sm">
        <CardContent className="pt-6">
          <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-text-muted" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question about Affectra AI..."
                aria-label="Ask a question about Affectra AI"
                className="w-full h-12 pl-10 pr-4 rounded-lg border border-border focus:border-success focus:ring-1 focus:ring-success outline-none transition-all"
                disabled={loading}
              />
            </div>
            <Button 
              type="submit" 
              className="w-full sm:w-auto h-12 px-8 bg-success hover:bg-success/90 text-white focus:ring-2 focus:ring-success/50 focus:outline-none"
              disabled={!query.trim() || loading}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  Searching...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  Search Knowledge
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
          <Card className="border-success/20 overflow-hidden">
            <div className="h-1 w-full bg-gradient-to-r from-success to-emerald-300"></div>
            <CardHeader className="bg-success-light/30 pb-4">
              <CardTitle className="text-xl flex items-center gap-2">
                <BrainIcon /> Generated Answer
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              <p className="text-text-main leading-relaxed whitespace-pre-wrap">{result.answer}</p>
            </CardContent>
          </Card>

          {result.retrieved_context.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-text-main flex items-center gap-2 px-1">
                <DatabaseIcon /> Retrieved Context
              </h3>
              <div className="grid gap-4">
                {result.retrieved_context.map((ctx, idx) => (
                  <Card key={idx} className="bg-background/50 text-sm">
                    <CardHeader className="py-3 px-4 border-b border-border bg-white flex flex-row items-center justify-between">
                      <div className="font-medium text-text-main truncate max-w-[70%]">
                        {ctx.source} <span className="text-text-muted font-normal text-xs ml-2">(Chunk {ctx.chunk_index})</span>
                      </div>
                      <div className="text-xs font-semibold px-2 py-1 bg-success-light text-success rounded-full">
                        Score: {ctx.score.toFixed(3)}
                      </div>
                    </CardHeader>
                    <CardContent className="p-4 text-text-muted leading-relaxed">
                      {ctx.text}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function BrainIcon() {
  return <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-success"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/></svg>
}

function DatabaseIcon() {
  return <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-text-muted"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>
}
