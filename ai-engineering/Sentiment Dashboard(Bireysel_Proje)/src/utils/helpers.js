export function getSentimentMeta(duygu) {
  switch (duygu) {
    case 'Pozitif':
      return { color: 'var(--positive)', bg: 'var(--positive-bg)', icon: '↑', label: 'Pozitif' }
    case 'Negatif':
      return { color: 'var(--negative)', bg: 'var(--negative-bg)', icon: '↓', label: 'Negatif' }
    default:
      return { color: 'var(--neutral)', bg: 'var(--neutral-bg)', icon: '–', label: 'Nötr' }
  }
}

export function formatDate(iso) {
  return new Date(iso).toLocaleString('tr-TR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

export function calcStats(analyses) {
  if (!analyses.length) return { avg: 0, positive: 0, negative: 0, neutral: 0 }
  const avg = analyses.reduce((s, a) => s + a.skor, 0) / analyses.length
  return {
    avg: Math.round(avg * 10) / 10,
    positive: analyses.filter(a => a.duygu === 'Pozitif').length,
    negative: analyses.filter(a => a.duygu === 'Negatif').length,
    neutral: analyses.filter(a => a.duygu === 'Nötr').length,
  }
}

export function getScoreLabel(skor) {
  if (skor >= 8) return 'Çok Pozitif'
  if (skor >= 6) return 'Pozitif'
  if (skor === 5) return 'Nötr'
  if (skor >= 3) return 'Negatif'
  return 'Çok Negatif'
}

export function saveToStorage(analyses) {
  try {
    localStorage.setItem('sentiment_history', JSON.stringify(analyses.slice(0, 50)))
  } catch { /* ignore */ }
}

export function loadFromStorage() {
  try {
    return JSON.parse(localStorage.getItem('sentiment_history') || '[]')
  } catch { return [] }
}
