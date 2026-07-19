"""
╔═══════════════════════════════════════════════════════════════╗
║                  TRANSFORMER MİMARİSİ                        ║
║                  SIFIRDAN PYTHON + NUMPY                     ║
║                      EĞİTİM AMAÇLI                           ║
╚═══════════════════════════════════════════════════════════════╝

BU KOD NE YAPAR?
================
Bu yazılım, Transformer yapısının temel bileşenlerini sıfırdan 
Python ve NumPy ile oluşturmaktadır. Adım adım şu işlemleri gerçekleştirir:

1. TOKENIZATION & EMBEDDING
   - Kelimeleri sayılara dönüştürme
   - Kelimelerin vektör temsillerini oluşturma
   - Pozisyon bilgisini ekleme

2. ATTENTION MEKANİZMASI
   - Self-Attention: Kelimelerin birbirleriyle ilişkisini bulma
   - Multi-Head Attention: Farklı perspektiflerden bakma
   
3. FEED-FORWARD AĞLAR
   - Non-lineer dönüşümler
   - Öğrenme kapasitesini artırma

4. NORMALIZASYON & RESIDUAL BAĞLANTILARI
   - Ağı stabil tutma
   - Gradyan akışını kolaylaştırma

5. SONU TARAFINDA SINIFLANDIRMA
   - Cümle temsilini sınıfa atama
"""

import numpy as np
import sys

# ═════════════════════════════════════════════════════════════════
# ÇIKTIYI DOSYAYA KAYDET
# ═════════════════════════════════════════════════════════════════
# Tüm çıktılar hem ekrana hem de dosyaya yazılacak

output_file = open('transformer_output.txt', 'w', encoding='utf-8')

class OutputRedirector:
    """Stdout'u hem ekrana hem dosyaya yönlendir"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Çıktıyı yönlendir
sys.stdout = OutputRedirector(sys.__stdout__, output_file)

# Konsol çıktısını düzenlemek için
np.set_printoptions(precision=3, suppress=True)

# ═════════════════════════════════════════════════════════════════
# 1. TEMEL UTILITY FONKSİYONLARI
# ═════════════════════════════════════════════════════════════════

def softmax(x):
    """
    Softmax Fonksiyonu
    ─────────────────
    Çıktıyı 0-1 arasında normalize eder ve olasılık dağılımı sağlar.
    Sayısal stabilite için max çıkarılır.
    """
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def relu(x):
    """
    ReLU Aktivasyon Fonksiyonu
    ────────────────────────
    f(x) = max(0, x) - Negatif değerleri sıfırla
    Non-lineer bir dönüşüm sağlar.
    """
    return np.maximum(0, x)


def layer_norm(x, eps=1e-6):
    """
    Layer Normalization (Katman Normalizasyonu)
    ──────────────────────────────────────────
    Her satırı ortalama=0, std=1 olacak şekilde normalize eder.
    Ağın eğitimini stabilize eder.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


# ═════════════════════════════════════════════════════════════════
# 2. BÖLÜM: TOKENIZATION & EMBEDDING
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("│ BÖLÜM 1: TOKENIZATION & EMBEDDING                           │")
print("="*70)

print("\n▸ TOKENIZATION (Kelime → Sayı Dönüşümü)")
print("─" * 70)
print("""
Transformer'ın ilk adımı: Metni sayısal forma çevirmek.
Her kelime bir ID numarası alıyor.
""")

sentence = ["Ben", "bugün", "okula", "gidiyorum"]
vocab = {"Ben": 0, "bugün": 1, "okula": 2, "gidiyorum": 3}
tokens = np.array([vocab[w] for w in sentence])

print(f"Orijinal cümle: {sentence}")
print(f"Token ID'leri:  {tokens}")

print("\n▸ EMBEDDING (Sayılar → Vektör)")
print("─" * 70)
print("""
Her kelime (token), bir vektör (sayı listesi) ile temsil edilir.
Semantik bilgi taşıyan 8 boyutlu vektörler oluşturuyoruz.
""")

vocab_size = len(vocab)
embedding_dim = 8
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

print(f"Embedding Matrisi şekli: {embedding_matrix.shape}")
print(f"Her kelime {embedding_dim} boyutlu vektörle temsil edilir")

X = embedding_matrix[tokens]
print(f"\nEmbedding sonucu şekli: {X.shape}")

print("\n▸ POSITIONAL ENCODING (Pozisyon Bilgisi)")
print("─" * 70)
print("""
Transformer'da kelime sırası önemlidir (farklı sıralamalara 
farklı anlamlar verilir). Her kelimeye pozisyon bilgisi eklenir.
""")

sequence_length = len(tokens)
position_embedding = np.random.randn(sequence_length, embedding_dim)
X = X + position_embedding  # Embedding + Pozisyon bilgisi

print(f"Pozisyonlar: {np.arange(sequence_length)}")
print(f"Nihai embedding şekli: {X.shape}")


# ═════════════════════════════════════════════════════════════════
# 3. BÖLÜM: SELF-ATTENTION MEKANİZMASI
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("│ BÖLÜM 2: SELF-ATTENTION MEKANİZMASI                         │")
print("="*70)

print("\n▸ QUERY, KEY, VALUE (Q, K, V)")
print("─" * 70)
print("""
Self-Attention 3 farklı lineer dönüşüm kullanır:
  • Query (Q): "Hangi kelimeyi arıyorum?"
  • Key (K):   "Ben ne kelimeyim?"
  • Value (V): "Benim anlamsal bilgim"

Her kelime bu 3 forma dönüştürülür.
""")

WQ = np.random.randn(embedding_dim, embedding_dim)
WK = np.random.randn(embedding_dim, embedding_dim)
WV = np.random.randn(embedding_dim, embedding_dim)

Q = X @ WQ
K = X @ WK
V = X @ WV

print(f"Query şekli:  {Q.shape}")
print(f"Key şekli:    {K.shape}")
print(f"Value şekli:  {V.shape}")

print("\n▸ ATTENTION SKORLARI (Q @ K.T)")
print("─" * 70)
print("""
Attention skorları = Query @ Key Transpose
Bu işlem, hangi kelimeye ne kadar dikkat etmesi gerektiğini söyler.
Yüksek skor = çok ilgili, düşük skor = az ilgili
""")

scores = Q @ K.T
print(f"Skorlar şekli: {scores.shape}")
print(f"Skor matrisi:\n{scores}")

print("\n▸ SCALED DOT PRODUCT (Ölçeklendirme)")
print("─" * 70)
print("""
Skorlar çok büyük olabilir. Sqrt(embedding_dim) ile bölünerek
softmax'ın gradyanları stabil hale getirilir.
""")

dk = K.shape[-1]
scaled_scores = scores / np.sqrt(dk)
print(f"Ölçeklendirilmiş skorlar:\n{scaled_scores}")

print("\n▸ ATTENTION AĞIRLIKLARI (Softmax)")
print("─" * 70)
print("""
Softmax ile skorlar olasılık dağılımına dönüştürülür.
Tüm ağırlıklar 0-1 arasında ve toplamı 1'dir.
""")

attention_weights = softmax(scaled_scores)
print(f"Attention ağırlıkları:\n{attention_weights}")

print("\n▸ FINAL ATTENTION ÇIKIŞI (Weights @ V)")
print("─" * 70)
print("""
Attention ağırlıkları, Value vektörleriyle çarpılır.
Sonuç: Diğer kelimelerin bilgisini ağırlıklı ortalama ile alırız.
""")

attention_output = attention_weights @ V
print(f"Attention çıkışı şekli: {attention_output.shape}")


# ─────────────────────────────────────────────────────────────────
# Self-Attention'u Fonksiyon Olarak Tanımlama
# ─────────────────────────────────────────────────────────────────

def self_attention(X, WQ, WK, WV):
    """
    Self-Attention Fonksiyonu
    ─────────────────────
    Tüm attention adımlarını bir fonksiyonda toplar.
    
    Girdi:
      X:  Embedding vektörleri
      WQ, WK, WV: Ağırlık matrisleri
      
    Çıktı:
      Attention çıkışı (X ile aynı boyutlar)
    """
    Q = X @ WQ
    K = X @ WK
    V = X @ WV
    
    scores = Q @ K.T
    scores = scores / np.sqrt(K.shape[-1])
    weights = softmax(scores)
    output = weights @ V
    
    return output


# ═════════════════════════════════════════════════════════════════
# 4. BÖLÜM: MULTI-HEAD ATTENTION
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("│ BÖLÜM 3: MULTI-HEAD ATTENTION                               │")
print("="*70)

print("\n▸ NİYE MULTI-HEAD?")
print("─" * 70)
print("""
Bir attention başı yerine birden fazla başı (head) kullanmak:
  • Farklı görevler için farklı representation öğrenme
  • Head-1: Kelimeler arasında sözdizim ilişkisi
  • Head-2: Semantik ilişki
  • vb...

Misal olarak 2 head kullanalım.
""")

num_heads = 2
head_dim = embedding_dim

# Head-1 ve Head-2 için ayrı ağırlıklar
WQ1 = np.random.randn(embedding_dim, head_dim)
WK1 = np.random.randn(embedding_dim, head_dim)
WV1 = np.random.randn(embedding_dim, head_dim)

WQ2 = np.random.randn(embedding_dim, head_dim)
WK2 = np.random.randn(embedding_dim, head_dim)
WV2 = np.random.randn(embedding_dim, head_dim)

print(f"Head sayısı: {num_heads}")
print(f"Her head boyutu: {head_dim}")

head1 = self_attention(X, WQ1, WK1, WV1)
head2 = self_attention(X, WQ2, WK2, WV2)

print(f"\nHead-1 çıkışı şekli: {head1.shape}")
print(f"Head-2 çıkışı şekli: {head2.shape}")

print("\n▸ HEADLERİN BİRLEŞTİRİLMESİ (Concatenation)")
print("─" * 70)
print("""
Tüm head çıktıları yan yana birleştirilir.
Bu, farklı perspektifleri aynı vektörde toplar.
""")

multi_head = np.concatenate([head1, head2], axis=-1)
print(f"Birleştirilmiş çıkış şekli: {multi_head.shape}")

print("\n▸ OUTPUT PROJECTION (Çıktı Dönüşümü)")
print("─" * 70)
print("""
Birleştirilmiş çıktı, tekrar orijinal boyuta geri döndürülür.
Bu, başlar arası etkileşimi öğrenmek için kullanılır.
""")

WO = np.random.randn(multi_head.shape[1], embedding_dim)
attention_output = multi_head @ WO
print(f"Final multi-head attention çıkışı: {attention_output.shape}")


# ═════════════════════════════════════════════════════════════════
# 5. BÖLÜM: FEED-FORWARD & NORMALIZATION
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("│ BÖLÜM 4: FEED-FORWARD VE NORMALIZASYON                      │")
print("="*70)

print("\n▸ RESIDUAL CONNECTION (Kalıntı Bağlantısı)")
print("─" * 70)
print("""
Attention çıkışı orijinal input ile toplanır.
Bu, bilgiyi "atlamaya" izin verir ve gradyan akışını iyileştirir.
""")

residual1 = attention_output + X
print(f"Residual bağlantı çıkışı: {residual1.shape}")

print("\n▸ LAYER NORMALIZATION")
print("─" * 70)
print("""
Çıktı normalize edilir (ortalama=0, std=1).
Ağın eğitimini stabilize eder ve daha hızlı yakınsamasını sağlar.
""")

normalized1 = layer_norm(residual1)

print("\n▸ FEED-FORWARD NETWORK (İleri Beslemeli Ağ)")
print("─" * 70)
print("""
Feed-forward ağ: 2 lineer katman + ReLU aktivasyon
  1. Geniş gizli katman (32 nöron)
  2. Tekrar orijinal boyuta inme

Denklem: FFN(x) = W2 * ReLU(W1 * x + b1) + b2

Bu, non-lineer dönüşüm ve öğrenme kapasitesi ekler.
""")

hidden_dim = 32

W1 = np.random.randn(embedding_dim, hidden_dim)
b1 = np.random.randn(hidden_dim)

W2 = np.random.randn(hidden_dim, embedding_dim)
b2 = np.random.randn(embedding_dim)

# Gizli katman ve ReLU
hidden = relu(normalized1 @ W1 + b1)
print(f"Gizli katman çıkışı: {hidden.shape}")

# Çıktı katmanı
ffn_output = hidden @ W2 + b2
print(f"FFN çıkışı: {ffn_output.shape}")

print("\n▸ RESIDUAL + LAYER NORM (Tekrar)")
print("─" * 70)
print("""
FFN çıkışı tekrar residual bağlantısıyla orijinal ile toplanır
ve normalize edilir. Bu, bir Transformer encoder bloğunun tamamıdır.
""")

encoder_output = layer_norm(normalized1 + ffn_output)
print(f"Encoder bloğu çıkışı: {encoder_output.shape}")


# ─────────────────────────────────────────────────────────────────
# Encoder Bloğunu Fonksiyon Olarak Tanımlama
# ─────────────────────────────────────────────────────────────────

def encoder_block(X, WQ1, WK1, WV1, WQ2, WK2, WV2, WO, W1, b1, W2, b2):
    """
    Encoder Bloğu
    ─────────
    Bir Transformer encoder bloğunun tamamı.
    
    Adımlar:
      1. Multi-Head Self-Attention (2 head)
      2. Residual bağlantı + Layer Norm
      3. Feed-Forward Network
      4. Residual bağlantı + Layer Norm
    """
    # Self-attention
    h1 = self_attention(X, WQ1, WK1, WV1)
    h2 = self_attention(X, WQ2, WK2, WV2)
    multi = np.concatenate([h1, h2], axis=-1)
    out = multi @ WO
    
    # Residual + Norm
    out = out + X
    out = layer_norm(out)
    
    # Feed-forward
    hidden = relu(out @ W1 + b1)
    out2 = hidden @ W2 + b2
    
    # Residual + Norm
    out = layer_norm(out + out2)
    
    return out


# ═════════════════════════════════════════════════════════════════
# 6. BÖLÜM: STACKLı ENCODER (TRANSFORMER MİMARİSİ)
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("│ BÖLÜM 5: STACKED ENCODER (TRANSFORMER)                      │")
print("="*70)

print("\n▸ NEDEN STACKED ENCODER?")
print("─" * 70)
print("""
Bir encoder bloğu kendi başına iyi değildir. Birden fazla blok
üst üste konularak (stack) derin bir ağ oluşturulur.
Her katman daha soyut temsiller öğrenir.

Misal: 2 encoder bloğu kullanalım.
""")

X_input = X  # İlk embedding
encoder1 = encoder_block(X_input, WQ1, WK1, WV1, WQ2, WK2, WV2, WO, W1, b1, W2, b2)
print(f"Encoder 1 çıkışı: {encoder1.shape}")

encoder2 = encoder_block(encoder1, WQ1, WK1, WV1, WQ2, WK2, WV2, WO, W1, b1, W2, b2)
print(f"Encoder 2 çıkışı: {encoder2.shape}")

print("\n▸ GLOBAL AVERAGE POOLING")
print("─" * 70)
print("""
Sequence çıkışını (4 kelime × 8 boyut) tek bir vektöre dönüştür.
Ortalama alarak, tüm kelimelerin bilgisini bir vektörde topla.
""")

pooled = np.mean(encoder2, axis=0)
print(f"Pooled çıkışı şekli: {pooled.shape}")
print(f"Çıktı: {pooled}")


# ═════════════════════════════════════════════════════════════════
# 7. BÖLÜM: SINIFLANDIRMA KATMANI
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("│ BÖLÜM 6: SINIFLANDIRMA KATMANI                              │")
print("="*70)

print("\n▸ CLASSIFICATION LAYER")
print("─" * 70)
print("""
Cümle temsili (pooled vektörü) sınıflara dönüştürülür.
İkileme sınıflandırma için 2 olasılık hesaplanır (ex: pozitif/negatif).
""")

num_classes = 2
W_classifier = np.random.randn(embedding_dim, num_classes)
b_classifier = np.random.randn(num_classes)

logits = pooled @ W_classifier + b_classifier
print(f"Logits: {logits}")

print("\n▸ SOFTMAX → OLASIKLIKLAR")
print("─" * 70)

prediction = softmax(logits)
print(f"Olasılıklar:\n  Sınıf 0: {prediction[0]:.4f}")
print(f"  Sınıf 1: {prediction[1]:.4f}")

predicted_class = np.argmax(prediction)
print(f"\nTahmin edilen sınıf: {predicted_class}")


# ═════════════════════════════════════════════════════════════════
# 8. BÖLÜM: YENİ CÜMLE TERİ (INFERENCE)
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("│ BÖLÜM 7: YENİ CÜMLE TESTİ (INFERENCE)                       │")
print("="*70)

print("\n▸ YENİ CÜMLE")
print("─" * 70)

new_sentence = ["Ben", "okula", "gidiyorum", "bugün"]
new_tokens = np.array([vocab[word] for word in new_sentence])

print(f"Cümle: {new_sentence}")
print(f"Tokens: {new_tokens}")

# Embedding
new_X = embedding_matrix[new_tokens] + position_embedding

# Transformer geçişleri
new_encoder = encoder_block(new_X, WQ1, WK1, WV1, WQ2, WK2, WV2, WO, W1, b1, W2, b2)
new_encoder = encoder_block(new_encoder, WQ1, WK1, WV1, WQ2, WK2, WV2, WO, W1, b1, W2, b2)

# Pooling
pooled_new = np.mean(new_encoder, axis=0)

# Sınıflandırma
logits_new = pooled_new @ W_classifier + b_classifier
prediction_new = softmax(logits_new)

print(f"\nTahmin:\n  Sınıf 0: {prediction_new[0]:.4f}")
print(f"  Sınıf 1: {prediction_new[1]:.4f}")
print(f"Sonuç: Sınıf {np.argmax(prediction_new)}")


# ═════════════════════════════════════════════════════════════════
# 9. BÖLÜM: ATTENTION GÖRSELLEŞTIRMESI
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("│ BÖLÜM 8: ATTENTION MAKİNESİ GÖRSELLEŞTIRMESI               │")
print("="*70)

print("\n▸ ATTENTION ÖSYESİ")
print("─" * 70)
print("""
Transformer hangi kelimeyi dinlediğini görelim.
Attention matrisi tüm kelime çiftleri arasındaki bağlantıyı gösterir.
""")

Q_viz = X_input @ WQ1
K_viz = X_input @ WK1

scores_viz = Q_viz @ K_viz.T
scores_viz = scores_viz / np.sqrt(embedding_dim)
weights_viz = softmax(scores_viz)

print("\nAttention matrisi:")
print(f"Boyut: {weights_viz.shape} (her satır bir kelime)")
print(weights_viz)

print("\n▸ HER KELIMEDE DIKKAT DAĞILIMI")
print("─" * 70)

for i, word in enumerate(sentence):
    print(f"\n'{word}' kelimesi şunlara dikkat ediyor:")
    for j, other in enumerate(sentence):
        attention_score = weights_viz[i, j]
        bar = "█" * int(attention_score * 30)
        print(f"  → '{other}': {attention_score:.3f} {bar}")

print("\n▸ EN YÜKSEK ATTENTION BAĞLANTILARI")
print("─" * 70)

for i in range(len(sentence)):
    highest_idx = np.argmax(weights_viz[i])
    print(f"'{sentence[i]}' en çok '{sentence[highest_idx]}'a dikkat ediyor")

# ═════════════════════════════════════════════════════════════════
# ÖZET
# ═════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("│ SONUÇ                                                        │")
print("="*70)
print("""
Başarıyla bir Transformer modeli oluşturduk ve çalıştırdık!

Öğrendiğimiz kavramlar:
  ✓ Tokenization & Embedding
  ✓ Self-Attention mekanizması
  ✓ Multi-Head Attention
  ✓ Feed-Forward Networks
  ✓ Residual Bağlantıları ve Layer Normalization
  ✓ Stacked Encoders
  ✓ Sınıflandırma

Bu minik Transformer, çalışma mantığını göstermektedir.
Gerçek modeller binlerce parametre ve katmanla çok daha karmaşıktır.

🎯 Sonraki adım: Bu kodu framework'ler (PyTorch, TensorFlow) 
ile ölçeklendirmek ve gerçek veri üzerinde eğitmektir.
""")
print("="*70)

# ═════════════════════════════════════════════════════════════════
# DOSYAYI KAPAT VE BİTİR
# ═════════════════════════════════════════════════════════════════

print("\n📄 Çıktılar 'transformer_output.txt' dosyasına kaydedilmiştir.")
sys.stdout = sys.__stdout__  # Çıktıyı normale döndür
output_file.close()

print("\n✅ Program tamamlandı!")
print("📁 Dosya: c:\\Users\\ASUS\\OneDrive\\Masaüstü\\NLP\\transformer_output.txt")


""" ======================================================================
│ BÖLÜM 1: TOKENIZATION & EMBEDDING                           │
======================================================================

▸ TOKENIZATION (Kelime → Sayı Dönüşümü)
──────────────────────────────────────────────────────────────────────

Transformer'ın ilk adımı: Metni sayısal forma çevirmek.
Her kelime bir ID numarası alıyor.

Orijinal cümle: ['Ben', 'bugün', 'okula', 'gidiyorum']
Token ID'leri:  [0 1 2 3]

▸ EMBEDDING (Sayılar → Vektör)
──────────────────────────────────────────────────────────────────────

Her kelime (token), bir vektör (sayı listesi) ile temsil edilir.
Semantik bilgi taşıyan 8 boyutlu vektörler oluşturuyoruz.

Embedding Matrisi şekli: (4, 8)
Her kelime 8 boyutlu vektörle temsil edilir

Embedding sonucu şekli: (4, 8)

▸ POSITIONAL ENCODING (Pozisyon Bilgisi)
──────────────────────────────────────────────────────────────────────

Transformer'da kelime sırası önemlidir (farklı sıralamalara 
farklı anlamlar verilir). Her kelimeye pozisyon bilgisi eklenir.

Pozisyonlar: [0 1 2 3]
Nihai embedding şekli: (4, 8)

======================================================================
│ BÖLÜM 2: SELF-ATTENTION MEKANİZMASI                         │
======================================================================

... (Q, K, V hesaplamaları, Attention Skorları ve Softmax ağırlıkları) ...

======================================================================
│ BÖLÜM 3: MULTI-HEAD ATTENTION                               │
======================================================================

... (Head'lerin oluşturulması, birleştirilmesi ve projeksiyon adımları) ...

======================================================================
│ BÖLÜM 4: FEED-FORWARD VE NORMALIZASYON                      │
======================================================================

... (Residual bağlantılar, Layer Norm ve FFN hesaplamaları) ...

======================================================================
│ BÖLÜM 5: STACKED ENCODER (TRANSFORMER)                      │
======================================================================

Encoder 1 çıkışı: (4, 8)
Encoder 2 çıkışı: (4, 8)

▸ GLOBAL AVERAGE POOLING
──────────────────────────────────────────────────────────────────────
Pooled çıkışı şekli: (8,)
Çıktı: [x.xxx x.xxx ... x.xxx]

======================================================================
│ BÖLÜM 6: SINIFLANDIRMA KATMANI                              │
======================================================================

Logits: [x.xxx x.xxx]

▸ SOFTMAX → OLASIKLIKLAR
──────────────────────────────────────────────────────────────────────
Olasılıklar:
  Sınıf 0: 0.xxx
  Sınıf 1: 0.xxx

Tahmin edilen sınıf: x

======================================================================
│ BÖLÜM 7: YENİ CÜMLE TESTİ (INFERENCE)                       │
======================================================================

... (Yeni cümle için tahmin sonuçları) ...

======================================================================
│ BÖLÜM 8: ATTENTION MAKİNESİ GÖRSELLEŞTIRMESI               │
======================================================================

... (Her kelimenin diğerlerine olan dikkat yoğunluğu görselleştirmesi) ...

======================================================================
│ SONUÇ                                                        │
======================================================================
Başarıyla bir Transformer modeli oluşturduk ve çalıştırdık!
..."""