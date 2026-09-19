import React from 'react'
import { NavLink } from 'react-router-dom'
import { Home, Activity, BookOpen, MessageSquare, Info, Brain, X } from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { name: 'Home', path: '/', icon: Home },
  { name: 'Multimodal Analysis', path: '/analysis', icon: Activity },
  { name: 'Explanation', path: '/explanation', icon: Brain },
  { name: 'Knowledge (RAG)', path: '/knowledge', icon: BookOpen },
  { name: 'AI Agent', path: '/agent', icon: MessageSquare },
  { name: 'About', path: '/about', icon: Info },
]

interface SidebarProps {
  onClose?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ onClose }) => {
  return (
    <aside className="w-64 border-r border-border bg-background flex flex-col h-full flex-shrink-0">
      <div className="p-6">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary rounded-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-text-main leading-tight">Affectra AI</h1>
              <p className="text-[10px] text-text-muted uppercase tracking-wider font-semibold">More Human AI</p>
            </div>
          </div>
          {onClose && (
            <button onClick={onClose} className="p-1 lg:hidden text-text-muted hover:bg-white rounded-md">
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        <nav className="space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={() => onClose?.()}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary-light text-primary"
                    : "text-text-muted hover:text-text-main hover:bg-white"
                )
              }
            >
              <item.icon className="w-5 h-5" />
              {item.name}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="mt-auto p-6">
        <div className="mt-4 text-xs text-text-muted">
          <p>Affectra AI</p>
          <p>v1.0.0</p>
        </div>
      </div>
    </aside>
  )
}
