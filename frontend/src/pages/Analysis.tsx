import React, { useState, useRef } from 'react'
import {
  MessageSquare,
  Mic,
  Video,
  Layers,
  Sparkles,
  Upload,
  X,
  FileAudio,
  FileVideo,
  AlertCircle,
  CheckCircle2,
  Brain,
  RefreshCw
} from 'lucide-react'
import { apiService } from '@/services/api'
import { usePrediction } from '@/context/PredictionContext'
import type { PredictionResponse } from '@/types/api'

type TabType = 'text' | 'audio' | 'video' | 'multimodal'

const EMOTION_COLORS: Record<string, { bg: string; text: string; bar: string }> = {
  joy: { bg: 'bg-amber-500/10', text: 'text-amber-500 border-amber-500/30', bar: 'bg-amber-500' },
  neutral: { bg: 'bg-slate-500/10', text: 'text-slate-400 border-slate-500/30', bar: 'bg-slate-400' },
  surprise: { bg: 'bg-purple-500/10', text: 'text-purple-400 border-purple-500/30', bar: 'bg-purple-500' },
  anger: { bg: 'bg-red-500/10', text: 'text-red-500 border-red-500/30', bar: 'bg-red-500' },
  sadness: { bg: 'bg-blue-500/10', text: 'text-blue-400 border-blue-500/30', bar: 'bg-blue-500' },
  fear: { bg: 'bg-indigo-500/10', text: 'text-indigo-400 border-indigo-500/30', bar: 'bg-indigo-500' },
  disgust: { bg: 'bg-emerald-500/10', text: 'text-emerald-400 border-emerald-500/30', bar: 'bg-emerald-500' },
}

const SENTIMENT_COLORS: Record<string, { bg: string; text: string; bar: string }> = {
  positive: { bg: 'bg-emerald-500/10', text: 'text-emerald-400 border-emerald-500/30', bar: 'bg-emerald-500' },
  negative: { bg: 'bg-rose-500/10', text: 'text-rose-400 border-rose-500/30', bar: 'bg-rose-500' },
  neutral: { bg: 'bg-slate-500/10', text: 'text-slate-400 border-slate-500/30', bar: 'bg-slate-400' },
}

export const Analysis: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('text')
  const [textInput, setTextInput] = useState('')
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null)

  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { prediction, setPredictionResult, inputSummary } = usePrediction()

  const audioInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAudioFile(e.target.files[0])
      setError(null)
    }
  }

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setVideoFile(file)
      setVideoPreviewUrl(URL.createObjectURL(file))
      setError(null)
    }
  }

  const clearAudio = () => {
    setAudioFile(null)
    if (audioInputRef.current) audioInputRef.current.value = ''
  }

  const clearVideo = () => {
    setVideoFile(null)
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl)
      setVideoPreviewUrl(null)
    }
    if (videoInputRef.current) videoInputRef.current.value = ''
  }

  const sampleTexts = [
    { label: 'Joy / Positive', text: "I am absolutely thrilled and overjoyed with how amazingly this turned out!" },
    { label: 'Anger / Negative', text: "This is completely unacceptable and I am furious about how poorly this was handled!" },
    { label: 'Neutral', text: "The presentation has been scheduled for 2:00 PM in Conference Room B." },
  ]

  const isAnalyzeDisabled = () => {
    if (isLoading) return true
    if (activeTab === 'text') return !textInput.trim()
    if (activeTab === 'audio') return !audioFile
    if (activeTab === 'video') return !videoFile
    if (activeTab === 'multimodal') return !textInput.trim() && !audioFile && !videoFile
    return true
  }

  const handleAnalyze = async () => {
    setError(null)
    setIsLoading(true)

    try {
      const formData = new FormData()
      const modalitiesUsed: ('text' | 'audio' | 'video')[] = []

      if ((activeTab === 'text' || activeTab === 'multimodal') && textInput.trim()) {
        formData.append('text', textInput.trim())
        modalitiesUsed.push('text')
      }

      if ((activeTab === 'audio' || activeTab === 'multimodal') && audioFile) {
        formData.append('audio_file', audioFile)
        modalitiesUsed.push('audio')
      }

      if ((activeTab === 'video' || activeTab === 'multimodal') && videoFile) {
        formData.append('video_file', videoFile)
        modalitiesUsed.push('video')
      }

      if (modalitiesUsed.length === 0) {
        throw new Error('Please provide at least one input (text, audio, or video) to analyze.')
      }

      const res: PredictionResponse = await apiService.predictRaw(formData)

      setPredictionResult(res, {
        text: textInput.trim() || undefined,
        audioName: audioFile?.name,
        videoName: videoFile?.name,
        modalitiesUsed,
        timestamp: new Date().toLocaleTimeString(),
      })
    } catch (err: any) {
      console.error('Prediction Error:', err)
      setError(err?.message || 'An unexpected error occurred during prediction.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-500 max-w-6xl mx-auto pb-12">
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2.5 rounded-2xl bg-primary/10 border border-primary/20 text-primary">
            <Sparkles className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-text-main tracking-tight">Multimodal Emotion Analysis</h1>
            <p className="text-text-muted text-sm mt-0.5">
              Powered by Affectra's Gated Multimodal Fusion Neural Network (DistilRoBERTa + Wav2Vec2 + ViT)
            </p>
          </div>
        </div>
      </div>

      {/* Main Grid: Inputs (Left) and Results (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Input Form (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="card p-6 border border-border bg-surface shadow-sm rounded-2xl space-y-6">
            {/* Tabs */}
            <div className="flex flex-wrap p-1 bg-background rounded-xl border border-border gap-1">
              <button
                type="button"
                onClick={() => setActiveTab('text')}
                className={`flex-1 min-w-[100px] flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'text'
                    ? 'bg-surface text-primary shadow-sm border border-border/80'
                    : 'text-text-muted hover:text-text-main hover:bg-surface/50'
                }`}
              >
                <MessageSquare className="w-4 h-4" />
                Text
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('audio')}
                className={`flex-1 min-w-[100px] flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'audio'
                    ? 'bg-surface text-primary shadow-sm border border-border/80'
                    : 'text-text-muted hover:text-text-main hover:bg-surface/50'
                }`}
              >
                <Mic className="w-4 h-4" />
                Audio
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('video')}
                className={`flex-1 min-w-[100px] flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'video'
                    ? 'bg-surface text-primary shadow-sm border border-border/80'
                    : 'text-text-muted hover:text-text-main hover:bg-surface/50'
                }`}
              >
                <Video className="w-4 h-4" />
                Video
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('multimodal')}
                className={`flex-1 min-w-[100px] flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-sm font-medium transition-all ${
                  activeTab === 'multimodal'
                    ? 'bg-primary text-white shadow-md'
                    : 'text-text-muted hover:text-text-main hover:bg-surface/50'
                }`}
              >
                <Layers className="w-4 h-4" />
                Fusion
              </button>
            </div>

            {/* Tab Contents */}
            <div className="space-y-4">
              {/* TEXT TAB */}
              {(activeTab === 'text' || activeTab === 'multimodal') && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-semibold text-text-main flex items-center gap-2">
                      <MessageSquare className="w-4 h-4 text-primary" />
                      Text Utterance Input
                    </label>
                    <span className="text-xs text-text-muted">{textInput.length} chars</span>
                  </div>
                  <textarea
                    rows={4}
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Enter text utterance to analyze emotions (e.g., 'I am so thrilled about this amazing result!')..."
                    className="w-full p-3.5 rounded-xl bg-background border border-border text-text-main placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary/50 text-sm transition-all resize-none"
                  />
                  {activeTab === 'text' && (
                    <div className="space-y-1.5">
                      <span className="text-xs font-medium text-text-muted">Sample prompts:</span>
                      <div className="flex flex-wrap gap-2">
                        {sampleTexts.map((s, idx) => (
                          <button
                            key={idx}
                            type="button"
                            onClick={() => setTextInput(s.text)}
                            className="text-xs px-2.5 py-1 rounded-lg bg-background border border-border hover:border-primary/40 text-text-muted hover:text-text-main transition-colors text-left"
                          >
                            {s.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* AUDIO TAB */}
              {(activeTab === 'audio' || activeTab === 'multimodal') && (
                <div className="space-y-3 pt-2">
                  <label className="text-sm font-semibold text-text-main flex items-center gap-2">
                    <Mic className="w-4 h-4 text-primary" />
                    Audio Recording (.wav, .mp3, .m4a)
                  </label>

                  <input
                    ref={audioInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={handleAudioChange}
                    className="hidden"
                    id="audio-upload"
                  />

                  {!audioFile ? (
                    <label
                      htmlFor="audio-upload"
                      className="flex flex-col items-center justify-center p-6 rounded-xl border-2 border-dashed border-border hover:border-primary/50 bg-background/50 hover:bg-background cursor-pointer transition-all text-center group"
                    >
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
                        <Upload className="w-5 h-5 text-primary" />
                      </div>
                      <span className="text-sm font-medium text-text-main">Click or drop audio file here</span>
                      <span className="text-xs text-text-muted mt-1">Supports WAV, MP3, M4A audio clips</span>
                    </label>
                  ) : (
                    <div className="flex items-center justify-between p-3.5 rounded-xl bg-primary/5 border border-primary/20">
                      <div className="flex items-center gap-3 truncate">
                        <FileAudio className="w-5 h-5 text-primary shrink-0" />
                        <div className="truncate">
                          <p className="text-sm font-medium text-text-main truncate">{audioFile.name}</p>
                          <p className="text-xs text-text-muted">{(audioFile.size / 1024).toFixed(1)} KB</p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={clearAudio}
                        className="p-1.5 rounded-lg hover:bg-background text-text-muted hover:text-text-main transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* VIDEO TAB */}
              {(activeTab === 'video' || activeTab === 'multimodal') && (
                <div className="space-y-3 pt-2">
                  <label className="text-sm font-semibold text-text-main flex items-center gap-2">
                    <Video className="w-4 h-4 text-primary" />
                    Video Clip (.mp4, .webm, .mov)
                  </label>

                  <input
                    ref={videoInputRef}
                    type="file"
                    accept="video/*"
                    onChange={handleVideoChange}
                    className="hidden"
                    id="video-upload"
                  />

                  {!videoFile ? (
                    <label
                      htmlFor="video-upload"
                      className="flex flex-col items-center justify-center p-6 rounded-xl border-2 border-dashed border-border hover:border-primary/50 bg-background/50 hover:bg-background cursor-pointer transition-all text-center group"
                    >
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
                        <Upload className="w-5 h-5 text-primary" />
                      </div>
                      <span className="text-sm font-medium text-text-main">Click or drop video file here</span>
                      <span className="text-xs text-text-muted mt-1">Supports MP4, WEBM, MOV video clips</span>
                    </label>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3.5 rounded-xl bg-primary/5 border border-primary/20">
                        <div className="flex items-center gap-3 truncate">
                          <FileVideo className="w-5 h-5 text-primary shrink-0" />
                          <div className="truncate">
                            <p className="text-sm font-medium text-text-main truncate">{videoFile.name}</p>
                            <p className="text-xs text-text-muted">{(videoFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={clearVideo}
                          className="p-1.5 rounded-lg hover:bg-background text-text-muted hover:text-text-main transition-colors"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                      {videoPreviewUrl && (
                        <div className="rounded-xl overflow-hidden border border-border bg-black max-h-48 flex items-center justify-center">
                          <video src={videoPreviewUrl} controls className="max-h-48 w-full object-contain" />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Error Banner */}
            {error && (
              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-start gap-3 animate-in fade-in">
                <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                <div className="flex-1 text-sm">{error}</div>
                <button type="button" onClick={() => setError(null)} className="text-rose-400 hover:text-rose-200">
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Submit Action */}
            <div className="pt-2">
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={isAnalyzeDisabled()}
                className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-primary to-primary-dark text-white font-semibold text-sm shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 focus:ring-2 focus:ring-primary/50 focus:outline-none"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Running Multimodal Neural Fusion...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Analyze Emotion & Sentiment
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Results Display (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {prediction ? (
            <div className="card p-6 border border-border bg-surface shadow-sm rounded-2xl space-y-6 animate-in slide-in-from-bottom-3 duration-500">
              {/* Results Top Header */}
              <div className="flex items-center justify-between border-b border-border/60 pb-4">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  <h2 className="text-lg font-bold text-text-main">Neural ML Output</h2>
                </div>
                {inputSummary && (
                  <div className="flex gap-1.5">
                    {inputSummary.modalitiesUsed.map((m) => (
                      <span key={m} className="px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20 text-[10px] uppercase font-bold tracking-wider">
                        {m}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* EMOTION RESULT */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-wider font-semibold text-text-muted">Predicted Emotion</span>
                  <span className="text-xs font-mono text-text-muted">
                    {((prediction.emotion.probabilities[prediction.emotion.label] || 0) * 100).toFixed(1)}% confidence
                  </span>
                </div>

                <div className="flex items-center justify-between p-3.5 rounded-xl border bg-background/50">
                  <span className="text-sm font-medium text-text-main capitalize">Emotion Category</span>
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border capitalize ${
                      EMOTION_COLORS[prediction.emotion.label.toLowerCase()]?.bg || 'bg-primary/10'
                    } ${EMOTION_COLORS[prediction.emotion.label.toLowerCase()]?.text || 'text-primary'}`}
                  >
                    {prediction.emotion.label}
                  </span>
                </div>

                {/* Emotion Probability Bars */}
                <div className="space-y-2 pt-1">
                  {Object.entries(prediction.emotion.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([emo, prob]) => {
                      const percentage = (prob * 100).toFixed(1)
                      const isTop = emo.toLowerCase() === prediction.emotion.label.toLowerCase()
                      const colors = EMOTION_COLORS[emo.toLowerCase()] || { bar: 'bg-primary' }

                      return (
                        <div key={emo} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className={`capitalize ${isTop ? 'font-bold text-text-main' : 'text-text-muted'}`}>
                              {emo}
                            </span>
                            <span className={`font-mono ${isTop ? 'font-bold text-text-main' : 'text-text-muted'}`}>
                              {percentage}%
                            </span>
                          </div>
                          <div className="w-full h-2 rounded-full bg-background overflow-hidden border border-border/40">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${colors.bar} ${
                                isTop ? 'opacity-100' : 'opacity-40'
                              }`}
                              style={{ width: `${Math.max(Number(percentage), 2)}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}
                </div>
              </div>

              {/* SENTIMENT RESULT */}
              <div className="space-y-3 pt-4 border-t border-border/60">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-wider font-semibold text-text-muted">Predicted Sentiment</span>
                  <span className="text-xs font-mono text-text-muted">
                    {((prediction.sentiment.probabilities[prediction.sentiment.label] || 0) * 100).toFixed(1)}% confidence
                  </span>
                </div>

                <div className="flex items-center justify-between p-3.5 rounded-xl border bg-background/50">
                  <span className="text-sm font-medium text-text-main capitalize">Sentiment Polarity</span>
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border capitalize ${
                      SENTIMENT_COLORS[prediction.sentiment.label.toLowerCase()]?.bg || 'bg-primary/10'
                    } ${SENTIMENT_COLORS[prediction.sentiment.label.toLowerCase()]?.text || 'text-primary'}`}
                  >
                    {prediction.sentiment.label}
                  </span>
                </div>

                {/* Sentiment Probability Bars */}
                <div className="space-y-2 pt-1">
                  {Object.entries(prediction.sentiment.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([sent, prob]) => {
                      const percentage = (prob * 100).toFixed(1)
                      const isTop = sent.toLowerCase() === prediction.sentiment.label.toLowerCase()
                      const colors = SENTIMENT_COLORS[sent.toLowerCase()] || { bar: 'bg-primary' }

                      return (
                        <div key={sent} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className={`capitalize ${isTop ? 'font-bold text-text-main' : 'text-text-muted'}`}>
                              {sent}
                            </span>
                            <span className={`font-mono ${isTop ? 'font-bold text-text-main' : 'text-text-muted'}`}>
                              {percentage}%
                            </span>
                          </div>
                          <div className="w-full h-2 rounded-full bg-background overflow-hidden border border-border/40">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${colors.bar} ${
                                isTop ? 'opacity-100' : 'opacity-40'
                              }`}
                              style={{ width: `${Math.max(Number(percentage), 2)}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}
                </div>
              </div>

              {/* State Saved Badge for Phase 6.4 */}
              <div className="p-3 rounded-xl bg-primary/5 border border-primary/20 text-xs text-primary flex items-center gap-2">
                <Brain className="w-4 h-4 shrink-0" />
                <span>Result saved in global state for Phase 6.4 Explainability, RAG & Agents.</span>
              </div>
            </div>
          ) : (
            <div className="card p-8 border border-border bg-surface/50 shadow-sm rounded-2xl flex flex-col items-center justify-center text-center h-full min-h-[380px]">
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4 text-primary">
                <Sparkles className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-bold text-text-main mb-1">Awaiting Prediction</h3>
              <p className="text-sm text-text-muted max-w-xs">
                Select your preferred input modality (Text, Audio, Video, or Fusion) and click Analyze to view real ML output.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
