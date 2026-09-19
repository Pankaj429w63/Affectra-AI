import React from 'react'
import { Info } from 'lucide-react'

const PlaceholderPage: React.FC<{
  title: string
  description: string
  icon: React.ElementType
  colorClass: string
  bgClass: string
}> = ({ title, description, icon: Icon, colorClass, bgClass }) => {
  return (
    <div className="flex flex-col items-center justify-center h-[60vh] text-center animate-in fade-in duration-500">
      <div className={`w-24 h-24 rounded-3xl flex items-center justify-center mb-8 shadow-sm ${bgClass}`}>
        <Icon className={`w-12 h-12 ${colorClass}`} />
      </div>
      <h1 className="text-3xl font-bold text-text-main mb-4">{title}</h1>
      <p className="text-lg text-text-muted max-w-lg mx-auto mb-8">
        {description}
      </p>
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-background border border-border text-sm font-medium text-text-muted">
        <span className="relative flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
          <span className="relative inline-flex rounded-full h-3 w-3 bg-primary"></span>
        </span>
        Coming in Phase 6.2+
      </div>
    </div>
  )
}


export const About: React.FC = () => (
  <PlaceholderPage
    title="About Affectra AI"
    description="A multimodal emotion and sentiment analysis system built for robust research and human-centric AI understanding."
    icon={Info}
    colorClass="text-text-muted"
    bgClass="bg-border"
  />
)
