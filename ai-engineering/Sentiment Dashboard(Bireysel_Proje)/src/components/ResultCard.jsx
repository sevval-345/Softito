import { useState } from 'react'
import { getSentimentMeta, getScoreLabel } from '../utils/helpers'
import styles from './ResultCard.module.css'

export default function ResultCard({ data }) {
  const [jsonOpen, setJsonOpen] = useState(false)
  if (!data) return null

  const meta = getSentimentMeta(data.duygu)
  const pct = (data.skor / 10) * 100

  const jsonOutput = JSON.stringify({
    yorum: data.yorum,
    duygu: data.duygu,
    skor: data.skor,
    anahtar_kelimeler: data.anahtar_kelimeler,
    kısa_özet: data.kisa_ozet,
  }, null, 2)

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <div className={styles.scoreCircle} style={{ background: meta.bg, color: meta.color }}>
          {data.skor}
        </div>
        <div className={styles.headerInfo}>
          <p className={styles.scoreLabel}>{getScoreLabel(data.skor)}</p>
          <div className={styles.barWrap}>
            <div
              className={styles.bar}
              style={{ width: `${pct}%`, background: meta.color }}
            />
          </div>
          <p className={styles.scoreNum}>{data.skor} / 10</p>
        </div>
        <span className={styles.badge} style={{ background: meta.bg, color: meta.color }}>
          {meta.icon} {data.duygu}
        </span>
      </div>

      <div className={styles.section}>
        <p className={styles.sectionLabel}>Kısa özet</p>
        <p className={styles.summary}>{data.kisa_ozet}</p>
      </div>

      <div className={styles.section}>
        <p className={styles.sectionLabel}>Anahtar kelimeler</p>
        <div className={styles.keywords}>
          {(data.anahtar_kelimeler || []).map(kw => (
            <span key={kw} className={styles.kw}>{kw}</span>
          ))}
        </div>
      </div>

      <div className={styles.jsonSection}>
        <button className={styles.jsonToggle} onClick={() => setJsonOpen(o => !o)}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
          </svg>
          {jsonOpen ? 'JSON çıktısını gizle' : 'JSON çıktısını göster'}
        </button>
        {jsonOpen && (
          <div className={styles.jsonBlock}>
            <button
              className={styles.copyBtn}
              onClick={() => navigator.clipboard.writeText(jsonOutput)}
            >
              Kopyala
            </button>
            <pre>{jsonOutput}</pre>
          </div>
        )}
      </div>
    </div>
  )
}
