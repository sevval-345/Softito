// Kelime listeleri
const POZITIF_KELIMELER = [
  'harika', 'mükemmel', 'süper', 'güzel', 'iyi', 'sevdim', 'beğendim', 'muhteşem',
  'enfes', 'memnun', 'mutlu', 'teşekkür', 'başarılı', 'kaliteli', 'hızlı', 'kolay',
  'rahat', 'şahane', 'olağanüstü', 'bravo', 'tebrik', 'tavsiye', 'love', 'great',
  'amazing', 'excellent', 'good', 'best', 'perfect', 'happy', 'thanks', 'awesome',
  'wonderful', 'fantastic', 'brilliant', 'impressed', 'recommend', 'satisfied',
  'pleased', 'enjoy', 'enjoyed', 'liked', 'nice', 'helpful', 'friendly', 'fast',
  'clean', 'worth', 'comfortable', 'professional', 'efficient', 'reliable'
]

const NEGATIF_KELIMELER = [
  'berbat', 'kötü', 'rezalet', 'korkunç', 'şikayet', 'bozuk', 'sorun', 'hata',
  'yavaş', 'pahalı', 'hayal kırıklığı', 'sinir', 'kızgın', 'üzgün', 'memnun değil',
  'tavsiye etmem', 'gelmeyeceğim', 'iade', 'şikayetçi', 'kandırmak', 'aldatmak',
  'beklentimi karşılamadı', 'vasat', 'yetersiz', 'ilgisiz', 'kabalık', 'uzun bekleme',
  'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointed', 'poor',
  'slow', 'expensive', 'broken', 'rude', 'never', 'avoid', 'waste', 'useless',
  'regret', 'complaint', 'problem', 'issue', 'failed', 'wrong', 'unhappy', 'angry',
  'frustrating', 'disgusting', 'overpriced', 'dirty', 'unreliable'
]

const NOTR_KELIMELER = [
  'tamam', 'fena değil', 'idare eder', 'ortalama', 'normal', 'standart',
  'okay', 'ok', 'average', 'fine', 'decent', 'alright', 'moderate'
]

// Anahtar kelime çıkarıcı
function extractKeywords(text) {
  const stopWords = new Set([
    'bir', 've', 'bu', 'da', 'de', 'ile', 'için', 'ama', 'fakat', 'çok',
    'daha', 'en', 'o', 'bu', 'şu', 'ne', 'ki', 'mi', 'mu', 'mı', 'mü',
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'it', 'its',
    'i', 'me', 'my', 'we', 'you', 'he', 'she', 'they', 'very', 'so',
    'that', 'this', 'have', 'has', 'had', 'do', 'did', 'not', 'no'
  ])

  const words = text
    .toLowerCase()
    .replace(/[^\w\sğüşıöçĞÜŞİÖÇ]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 3 && !stopWords.has(w))

  const freq = {}
  for (const w of words) freq[w] = (freq[w] || 0) + 1

  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([w]) => w)
}

// Skor hesaplayıcı
function calcScore(text) {
  const lower = text.toLowerCase()
  let pos = 0
  let neg = 0

  for (const w of POZITIF_KELIMELER) {
    if (lower.includes(w)) pos += w.split(' ').length > 1 ? 2 : 1
  }
  for (const w of NEGATIF_KELIMELER) {
    if (lower.includes(w)) neg += w.split(' ').length > 1 ? 2 : 1
  }

  // Noktalama sinyalleri
  const exclamations = (text.match(/!/g) || []).length
  const questions = (text.match(/\?/g) || []).length
  if (exclamations > 0 && pos > neg) pos += exclamations
  if (questions > 1) neg += 1

  // Büyük harf yoğunluğu (sinir işareti)
  const upperRatio = (text.match(/[A-ZĞÜŞİÖÇ]/g) || []).length / text.length
  if (upperRatio > 0.3 && neg > 0) neg += 2

  const total = pos + neg
  if (total === 0) {
    // Nötr kelime var mı
    for (const w of NOTR_KELIMELER) {
      if (lower.includes(w)) return 5
    }
    return 5
  }

  const ratio = pos / total
  if (ratio >= 0.7) return Math.min(10, 7 + Math.round(ratio * 3))
  if (ratio >= 0.5) return 6
  if (ratio >= 0.4) return 5
  if (ratio >= 0.25) return Math.max(3, 4 - Math.round((0.4 - ratio) * 5))
  return Math.max(1, 2 - (neg > 4 ? 1 : 0))
}

// Özet üretici
function generateSummary(text, duygu, skor, keywords) {
  const kw = keywords[0] || 'içerik'
  if (text.length < 10) return 'Yorum değerlendirmek için çok kısa.'

  if (duygu === 'Pozitif') {
    if (skor >= 9) return `Kullanıcı "${kw}" konusunda son derece olumlu bir deneyim aktarıyor.`
    return `Kullanıcı genel olarak memnun görünüyor; özellikle "${kw}" konusu öne çıkıyor.`
  }
  if (duygu === 'Negatif') {
    if (skor <= 2) return `Kullanıcı ciddi şikayetlerini dile getiriyor; "${kw}" konusu ön plana çıkıyor.`
    return `Kullanıcı bazı olumsuz deneyimler yaşamış, "${kw}" konusunda hayal kırıklığı göze çarpıyor.`
  }
  return `Yorum nötr bir ton taşıyor; kullanıcı "${kw}" hakkında dengeli bir değerlendirme yapıyor.`
}

// Gecikme simülasyonu (gerçekçilik için)
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export async function analyzeComment(yorum) {
  await delay(600 + Math.random() * 600)

  const trimmed = yorum.trim()

  if (trimmed.length < 3) {
    return {
      yorum: trimmed,
      duygu: 'Nötr',
      skor: 5,
      anahtar_kelimeler: [],
      kisa_ozet: 'Yorum değerlendirmek için çok kısa.',
    }
  }

  const skor = calcScore(trimmed)
  const duygu = skor >= 6 ? 'Pozitif' : skor <= 4 ? 'Negatif' : 'Nötr'
  const anahtar_kelimeler = extractKeywords(trimmed)
  const kisa_ozet = generateSummary(trimmed, duygu, skor, anahtar_kelimeler)

  return {
    yorum: trimmed,
    duygu,
    skor,
    anahtar_kelimeler,
    kisa_ozet,
  }
}
