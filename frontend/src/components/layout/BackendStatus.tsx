import React, { useEffect, useState } from 'react'
import { apiService } from '@/services/api'

export const BackendStatus: React.FC = () => {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline'>('checking')

  useEffect(() => {
    let mounted = true;
    apiService.checkHealth()
      .then(() => {
        if (mounted) setStatus('online')
      })
      .catch(() => {
        if (mounted) setStatus('offline')
      })
    return () => { mounted = false; }
  }, [])

  if (status === 'checking') {
    return (
      <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-background border border-border">
        <div className="w-2 h-2 rounded-full bg-text-muted animate-pulse"></div>
        <span className="text-sm font-medium text-text-muted">Checking backend...</span>
      </div>
    )
  }

  if (status === 'offline') {
    return (
      <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-error-light border border-error/20">
        <div className="w-2 h-2 rounded-full bg-error"></div>
        <span className="text-sm font-medium text-error">Backend Offline</span>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-success-light border border-success/20">
      <div className="w-2 h-2 rounded-full bg-success"></div>
      <span className="text-sm font-medium text-success">Backend Online</span>
    </div>
  )
}
