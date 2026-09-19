import React, { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { BackendStatus } from './BackendStatus'
import { Moon, Settings, Menu } from 'lucide-react'

export const AppShell: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      {/* Mobile Sidebar Overlay */}
      {mobileMenuOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}
      
      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 transform transition-transform duration-300 lg:relative lg:translate-x-0 ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <Sidebar onClose={() => setMobileMenuOpen(false)} />
      </div>

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="h-16 lg:h-20 border-b border-border bg-white flex items-center justify-between lg:justify-end px-4 lg:px-8 gap-4 flex-shrink-0">
          <div className="flex items-center lg:hidden">
            <button 
              className="p-2 -ml-2 text-text-muted hover:bg-background rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open Menu"
            >
              <Menu className="w-6 h-6" />
            </button>
          </div>

          <div className="flex items-center gap-2 lg:gap-4">
            <BackendStatus />
            <button className="w-10 h-10 hidden sm:flex items-center justify-center rounded-full hover:bg-background text-text-muted transition-colors focus:outline-none focus:ring-2 focus:ring-primary/50" aria-label="Toggle dark mode">
              <Moon className="w-5 h-5" />
            </button>
            <button className="w-10 h-10 hidden sm:flex items-center justify-center rounded-full hover:bg-background text-text-muted transition-colors focus:outline-none focus:ring-2 focus:ring-primary/50" aria-label="Settings">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
          <div className="max-w-6xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
