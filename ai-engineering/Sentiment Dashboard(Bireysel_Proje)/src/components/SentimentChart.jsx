import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, CartesianGrid, ReferenceLine,
} from 'recharts'
import styles from './SentimentChart.module.css'

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className={styles.tooltip}>
      <p className={styles.tooltipScore}>{d.skor} / 10</p>
      <p className={styles.tooltipDuygu}>{d.duygu}</p>
      <p className={styles.tooltipYorum}>{d.yorum?.substring(0, 60)}{d.yorum?.length > 60 ? '…' : ''}</p>
    </div>
  )
}

export default function SentimentChart({ analyses }) {
  if (analyses.length < 2) return null

  const data = [...analyses].reverse().map((a, i) => ({
    ...a,
    index: i + 1,
  }))

  return (
    <div className={styles.wrap}>
      <p className={styles.title}>Skor geçmişi</p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
          <XAxis dataKey="index" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickLine={false} axisLine={false} />
          <YAxis domain={[1, 10]} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} tickLine={false} axisLine={false} ticks={[1,3,5,7,10]} />
          <ReferenceLine y={5} stroke="rgba(255,255,255,0.07)" strokeDasharray="4 4" />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.07)' }} />
          <Line
            type="monotone"
            dataKey="skor"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={{ fill: 'var(--accent)', strokeWidth: 0, r: 3 }}
            activeDot={{ r: 5, fill: 'var(--accent)' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
