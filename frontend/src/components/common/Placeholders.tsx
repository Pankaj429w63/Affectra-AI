import React from 'react'
import { Loader2, AlertCircle, FileX } from 'lucide-react'

export const LoadingState: React.FC<{ message?: string }> = ({ message = 'Loading...' }) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-text-muted">
      <Loader2 className="w-8 h-8 animate-spin mb-4 text-primary" />
      <p className="text-sm">{message}</p>
    </div>
  )
}

export const ErrorState: React.FC<{ message?: string; title?: string }> = ({ 
  title = 'Something went wrong',
  message = 'An unexpected error occurred while processing your request.' 
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center">
      <div className="w-12 h-12 rounded-full bg-error-light flex items-center justify-center mb-4">
        <AlertCircle className="w-6 h-6 text-error" />
      </div>
      <h3 className="text-lg font-semibold text-text-main mb-2">{title}</h3>
      <p className="text-sm text-text-muted max-w-md">{message}</p>
    </div>
  )
}

export const EmptyState: React.FC<{ message?: string; title?: string }> = ({ 
  title = 'No Data Available',
  message = 'There is currently no data to display here.' 
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center border-2 border-dashed border-border rounded-xl bg-background">
      <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center mb-4 shadow-sm">
        <FileX className="w-6 h-6 text-text-muted" />
      </div>
      <h3 className="text-lg font-semibold text-text-main mb-2">{title}</h3>
      <p className="text-sm text-text-muted max-w-sm">{message}</p>
    </div>
  )
}
