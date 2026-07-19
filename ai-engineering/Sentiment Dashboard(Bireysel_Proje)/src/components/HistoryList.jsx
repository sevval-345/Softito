import { getSentimentMeta, formatDate } from '../utils/helpers'
import styles from './HistoryList.module.css'

export default function HistoryList({ analyses, onClear }) {
  if (!analyses.length) return null

  return (
    <div className={styles.wrap}>
      <div className={styles.header}>
        <p className={styles.title}>Geçmiş analizler</p>
        <button className={styles.clearBtn} onClick={onClear}>Temizle</button>
      </div>
      <div className={styles.list}>
        {analyses.map(a => {
          const meta = getSentimentMeta(a.duygu)
          return (
            <div key={a.id} className={styles.item}>
              <div className={styles.dot} style={{ background: meta.color }} />
              <div className={styles.content}>
                <p className={styles.yorum}>{a.yorum?.substring(0, 80)}{a.yorum?.length > 80 ? '…' : ''}</p>
                <p className={styles.meta}>
                  <span style={{ color: meta.color }}>{a.duygu}</span>
                  <span className={styles.sep}>·</span>
                  <span className={styles.mono}>{a.skor}/10</span>
                  <span className={styles.sep}>·</span>
                  <span>{formatDate(a.timestamp)}</span>
                </p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
