import { calcStats } from '../utils/helpers'
import styles from './StatsBar.module.css'

export default function StatsBar({ analyses }) {
  if (!analyses.length) return null
  const s = calcStats(analyses)

  const items = [
    { label: 'Toplam analiz', value: analyses.length, color: 'var(--text-primary)' },
    { label: 'Ortalama skor', value: s.avg.toFixed(1), color: 'var(--accent)', mono: true },
    { label: 'Pozitif', value: s.positive, color: 'var(--positive)' },
    { label: 'Nötr', value: s.neutral, color: 'var(--neutral)' },
    { label: 'Negatif', value: s.negative, color: 'var(--negative)' },
  ]

  return (
    <div className={styles.bar}>
      {items.map(item => (
        <div key={item.label} className={styles.item}>
          <span className={styles.value} style={{ color: item.color, fontFamily: item.mono ? 'var(--font-mono)' : undefined }}>
            {item.value}
          </span>
          <span className={styles.label}>{item.label}</span>
        </div>
      ))}
    </div>
  )
}
