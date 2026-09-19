/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#f8fafc', // slate-50
        card: '#ffffff',
        primary: {
          DEFAULT: '#2563eb', // blue-600
          hover: '#1d4ed8', // blue-700
          light: '#eff6ff', // blue-50
        },
        secondary: {
          DEFAULT: '#8b5cf6', // violet-500
          hover: '#7c3aed', // violet-600
          light: '#f5f3ff', // violet-50
        },
        success: {
          DEFAULT: '#10b981', // emerald-500
          light: '#ecfdf5', // emerald-50
        },
        warning: {
          DEFAULT: '#f59e0b', // amber-500
          light: '#fffbeb', // amber-50
        },
        error: {
          DEFAULT: '#ef4444', // red-500
          light: '#fef2f2', // red-50
        },
        text: {
          main: '#0f172a', // slate-900
          muted: '#64748b', // slate-500
        },
        border: '#e2e8f0', // slate-200
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05)',
      }
    },
  },
  plugins: [],
}
