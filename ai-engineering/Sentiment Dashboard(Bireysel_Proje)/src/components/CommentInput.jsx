import { useState, useRef } from 'react'
import styles from './CommentInput.module.css'

export default function CommentInput({ onAnalyze, loading }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)

  function handleSubmit() {
    if (!text.trim() || loading) return
    onAnalyze(text.trim())
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSubmit()
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.inputBox}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Analiz etmek istediğiniz yorumu buraya yazın..."
          rows={4}
          disabled={loading}
        />
        <div className={styles.footer}>
          <span className={styles.charCount}>{text.length} karakter · Ctrl+Enter ile analiz et</span>
          <button
            className={styles.btn}
            onClick={handleSubmit}
            disabled={!text.trim() || loading}
          >
            {loading ? (
              <span className={styles.spinner} />
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M22 2 11 13" /><path d="m22 2-7 20-4-9-9-4 20-7z" />
              </svg>
            )}
            {loading ? 'Analiz ediliyor...' : 'Analiz Et'}
          </button>
        </div>
      </div>
    </div>
  )
}
