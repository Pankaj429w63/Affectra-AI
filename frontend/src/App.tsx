import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { Dashboard } from './pages/Dashboard'
import { Analysis } from './pages/Analysis'
import { Knowledge } from './pages/Knowledge'
import { Agent } from './pages/Agent'
import { Explanation } from './pages/Explanation'
import { About } from './pages/Placeholders'
import { PredictionProvider } from './context/PredictionContext'

function App() {
  return (
    <PredictionProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AppShell />}>
            <Route index element={<Dashboard />} />
            <Route path="analysis" element={<Analysis />} />
            <Route path="explanation" element={<Explanation />} />
            <Route path="knowledge" element={<Knowledge />} />
            <Route path="agent" element={<Agent />} />
            <Route path="about" element={<About />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </PredictionProvider>
  )
}

export default App
