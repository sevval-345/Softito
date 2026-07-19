import { useState, useCallback } from 'react'
import { analyzeComment } from '../utils/api'
import { saveToStorage, loadFromStorage } from '../utils/helpers'

export function useAnalysis() {
  const [analyses, setAnalyses] = useState(() => loadFromStorage())
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [latest, setLatest] = useState(null)

  const analyze = useCallback(async (yorum) => {
    if (!yorum.trim()) return
    setLoading(true)
    setError(null)

    try {
      const result = await analyzeComment(yorum)
      const entry = { ...result, id: Date.now(), timestamp: new Date().toISOString() }
      setLatest(entry)
      setAnalyses(prev => {
        const next = [entry, ...prev]
        saveToStorage(next)
        return next
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const clearHistory = useCallback(() => {
    setAnalyses([])
    setLatest(null)
    localStorage.removeItem('sentiment_history')
  }, [])

  return { analyses, loading, error, latest, analyze, clearHistory }
}
