import { useAnalysis } from './hooks/useAnalysis'
import CommentInput from './components/CommentInput'
import ResultCard from './components/ResultCard'
import StatsBar from './components/StatsBar'
import SentimentChart from './components/SentimentChart'
import HistoryList from './components/HistoryList'
import ErrorBanner from './components/ErrorBanner'
import styles from './App.module.css'

export default function App() {
  const { analyses, loading, error, latest, analyze, clearHistory } = useAnalysis()

  return (
    <div className={styles.layout}>
      <header className={styles.header}>
        <div className={styles.logo}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          <span className={styles.logoText}>Sentiment Dashboard</span>
        </div>
        <span className={styles.version}>yerel analiz motoru</span>
      </header>

      <main className={styles.main}>
        <div className={styles.pane}>
          <h1 className={styles.pageTitle}>Yorum analizi</h1>
          <p className={styles.pageDesc}>Sosyal medya yorumlarını API gerektirmeden, yerel NLP motoru ile analiz edin.</p>

          <CommentInput onAnalyze={analyze} loading={loading} />
          <ErrorBanner message={error} />
          <ResultCard data={latest} />
        </div>

        <div className={styles.pane}>
          <StatsBar analyses={analyses} />
          <SentimentChart analyses={analyses} />
          <HistoryList analyses={analyses} onClear={clearHistory} />
        </div>
      </main>
    </div>
  )
}
