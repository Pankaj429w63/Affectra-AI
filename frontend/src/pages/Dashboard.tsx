import React from 'react'
import { Activity, Brain, BookOpen, MessageSquare, ArrowRight } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { useNavigate } from 'react-router-dom'

const features = [
  {
    title: 'Multimodal Analysis',
    description: 'Analyze emotions from text, audio, or video.',
    icon: Activity,
    path: '/analysis',
    color: 'text-primary',
    bg: 'bg-primary-light',
  },
  {
    title: 'Explain',
    description: 'Get detailed explanations for predictions.',
    icon: Brain,
    path: '/explanation',
    color: 'text-secondary',
    bg: 'bg-secondary-light',
  },
  {
    title: 'Knowledge (RAG)',
    description: 'Ask questions about Affectra AI.',
    icon: BookOpen,
    path: '/knowledge',
    color: 'text-success',
    bg: 'bg-success-light',
  },
  {
    title: 'AI Agent',
    description: 'Chat with an intelligent, safe AI agent.',
    icon: MessageSquare,
    path: '/agent',
    color: 'text-warning',
    bg: 'bg-warning-light',
  },
]

export const Dashboard: React.FC = () => {
  const navigate = useNavigate()

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary-light to-secondary-light rounded-3xl p-10 border border-primary/10 shadow-sm relative overflow-hidden">
        <div className="absolute right-0 top-0 w-1/3 h-full opacity-10 pointer-events-none">
          <Brain className="w-full h-full text-primary scale-150 translate-x-1/4 -translate-y-1/4" />
        </div>
        
        <div className="relative z-10 max-w-2xl">
          <h1 className="text-4xl font-extrabold text-text-main mb-4 tracking-tight">
            Human Emotions. <span className="text-primary">Deeper Understanding.</span>
          </h1>
          <p className="text-lg text-text-muted mb-8 font-medium">
            Analyze, explain, learn, and chat — all in one place.
          </p>
          
          <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white shadow-sm inline-block">
            <h2 className="text-xl font-bold text-text-main flex items-center gap-2 mb-2">
              <Brain className="w-6 h-6 text-primary" />
              Affectra AI
            </h2>
            <p className="text-text-muted">
              Multimodal Emotion Analysis with Explainable and Knowledge-Grounded AI.
              Upload text, audio, or video to analyze emotions, explore explanations, ask questions, or chat with our AI agent.
            </p>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature) => (
            <Card key={feature.title} className="group hover:shadow-md transition-shadow flex flex-col h-full cursor-pointer" onClick={() => navigate(feature.path)}>
              <CardHeader>
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${feature.bg}`}>
                  <feature.icon className={`w-6 h-6 ${feature.color}`} />
                </div>
                <CardTitle className="text-lg">{feature.title}</CardTitle>
                <CardDescription className="text-sm mt-2">{feature.description}</CardDescription>
              </CardHeader>
              <CardContent className="mt-auto pt-4">
                <Button variant="ghost" className="w-full justify-between group-hover:bg-background focus:ring-2 focus:ring-primary/50 focus:outline-none">
                  Explore
                  <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Trust Badges */}
      <section className="pt-8 border-t border-border grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-success-light flex items-center justify-center flex-shrink-0">
            <Activity className="w-5 h-5 text-success" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-text-main">Privacy First</h4>
            <p className="text-xs text-text-muted">Your data stays secure.</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-secondary-light flex items-center justify-center flex-shrink-0">
            <Brain className="w-5 h-5 text-secondary" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-text-main">Explainable AI</h4>
            <p className="text-xs text-text-muted">Understand the reasoning.</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-primary-light flex items-center justify-center flex-shrink-0">
            <BookOpen className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-text-main">Knowledge Grounded</h4>
            <p className="text-xs text-text-muted">Accurate RAG information.</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-warning-light flex items-center justify-center flex-shrink-0">
            <MessageSquare className="w-5 h-5 text-warning" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-text-main">Responsible AI</h4>
            <p className="text-xs text-text-muted">Safe, guardrailed agent.</p>
          </div>
        </div>
      </section>
    </div>
  )
}
