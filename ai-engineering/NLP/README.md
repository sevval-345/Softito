<div align="center">

# 📝 Natural Language Processing (NLP)

### Doğal Dil İşleme Algoritmaları, Metin Analizi ve Dil Temsili Çalışmaları

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F3B?style=for-the-badge)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Word2Vec](https://img.shields.io/badge/Word2Vec-4285F4?style=for-the-badge)
![TF--IDF](https://img.shields.io/badge/TF--IDF-4CAF50?style=for-the-badge)

Bu klasör, **Doğal Dil İşleme (Natural Language Processing - NLP)** alanındaki temel kavramlardan modern metin temsil yöntemlerine kadar uzanan uygulamalı çalışmaları içermektedir.

</div>

---

# 📖 Hakkında

Natural Language Processing (NLP), bilgisayarların insan dilini anlamasını, işlemesini ve yorumlamasını sağlayan yapay zekâ disiplinidir.

Bu klasörde yer alan notebooklar; ham metinlerin işlenmesi, temizlenmesi, sayısallaştırılması ve makine öğrenmesi modelleri tarafından kullanılabilir hale getirilmesi süreçlerini kapsamaktadır.

Projeler klasik NLP tekniklerinden başlayarak modern kelime gömme (Word Embedding) yöntemlerine kadar ilerleyen bir öğrenme akışı sunmaktadır.

---

# 🎯 Amaç

Bu klasörün temel amacı;

- Ham metin verilerini analiz etmek,
- Metin ön işleme (Text Preprocessing) adımlarını uygulamak,
- Kelime temsil yöntemlerini incelemek,
- NLP problemleri için özellik çıkarımı gerçekleştirmek,
- Makine öğrenmesi modellerine uygun veri hazırlama süreçlerini öğrenmektir.

---

# 📂 Klasör Yapısı

```text
NLP/
│
├── Text Preprocessing
├── Tokenization
├── Stop Words
├── Stemming
├── Lemmatization
├── Bag of Words
├── N-Gram
├── TF-IDF
├── Word2Vec
├── Text Vectorization
└── NLP Uygulamaları
```

---

# 📚 İçerik

Bu klasörde aşağıdaki NLP konularına yönelik çalışmalar bulunmaktadır.

## 📌 Metin Ön İşleme (Text Preprocessing)

Ham metinlerin analiz edilebilir hale getirilmesi amacıyla uygulanan temel işlemler.

- Text Cleaning
- Lowercase Conversion
- Noktalama İşaretlerinin Temizlenmesi
- Sayı Temizleme
- Stop Word Removal
- Tokenization
- Normalization

---

## 📌 Kelime Kökü Bulma

Kelime boyutunu azaltarak anlamsal benzerlik oluşturmayı amaçlayan yöntemler.

- Stemming
- Lemmatization

---

## 📌 Metin Temsili (Feature Extraction)

Metinlerin sayısal vektörlere dönüştürülmesi.

- Bag of Words (BoW)
- N-Gram
- TF-IDF
- Count Vectorizer

---

## 📌 Word Embedding

Kelimelerin anlamsal ilişkilerini öğrenen dağıtık temsil yöntemleri.

- Word2Vec
- Continuous Bag of Words (CBOW)
- Skip-Gram

---

## 📌 NLP Uygulamaları

Metin verileri üzerinde gerçekleştirilen temel uygulamalar.

- Text Classification
- Sentiment Analysis
- Feature Engineering
- Document Vectorization

---

# 🧠 Kullanılan NLP Pipeline

```text
Raw Text
    │
    ▼
Text Cleaning
    │
    ▼
Tokenization
    │
    ▼
Stop Word Removal
    │
    ▼
Stemming / Lemmatization
    │
    ▼
Feature Extraction
(BoW • TF-IDF • Word2Vec)
    │
    ▼
Machine Learning Model
    │
    ▼
Prediction
```

---

# 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|-----------|----------------|
| 🐍 Python | Programlama Dili |
| 📚 NLTK | Metin ön işleme |
| 🚀 spaCy | NLP Pipeline |
| 🤖 Scikit-Learn | TF-IDF, CountVectorizer ve ML modelleri |
| 📖 Word2Vec | Kelime gömme (Embedding) |
| 🔢 NumPy | Sayısal işlemler |
| 🐼 Pandas | Veri analizi |
| 📊 Matplotlib | Görselleştirme |
| 📈 Seaborn | İstatistiksel grafikler |

---

# 📌 Öğrenilen Konular

Bu klasör sonunda aşağıdaki NLP kavramları hakkında uygulamalı bilgi edinilebilir.

- Text Preprocessing
- Tokenization
- Stop Words
- Stemming
- Lemmatization
- Regular Expressions
- Bag of Words
- N-Gram Models
- TF-IDF
- Word Embedding
- Word2Vec
- Count Vectorizer
- Feature Engineering
- Text Classification
- Sentiment Analysis
- Document Representation

---

# 🚀 Kullanım

Projeyi klonlayın.

```bash
git clone https://github.com/sevval-345/Softito.git
```

İlgili dizine geçin.

```bash
cd ai-engineering/NLP
```

Gerekli kütüphaneleri yükleyin.

```bash
pip install -r requirements.txt
```

Notebook dosyalarını sırasıyla çalıştırarak NLP süreçlerini adım adım inceleyebilirsiniz.

---

# 💡 Not

Bu klasör, **LLM Engineering** çalışmalarının temelini oluşturan klasik NLP konularını kapsamaktadır. Modern Transformer tabanlı dil modellerine geçmeden önce metin ön işleme, özellik çıkarımı ve kelime temsili yöntemlerini anlamaya odaklanmaktadır. GitHub README'lerinde kısa, anlaşılır ve klasörün amacını net ifade eden yapılar önerilir; bu README de o yaklaşıma uygun hazırlanmıştır. :contentReference[oaicite:0]{index=0}

---

<div align="center">

⭐ Bu klasörü faydalı bulduysanız projeye yıldız vermeyi unutmayın.

**Geliştiren:** **Şevval Mıkçı**

</div>
