<div align="center">

# 🧠 LLM Engineering & Gelişmiş NLP Pipeline

### Büyük Dil Modellerini (LLM) Temellerden Üretim Ortamına Kadar Uçtan Uca Öğrenme Yolculuğu

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

**Transformer • Fine-Tuning • Quantization • RLHF • Mikroservis Mimarisi**

</div>

---

# 📖 Proje Hakkında

Bu depo, **Modern Doğal Dil İşleme (NLP)** ve **Büyük Dil Modelleri (LLM)** üzerine hazırlanmış kapsamlı bir mühendislik çalışmasıdır.

Amaç yalnızca bir modeli eğitmek değil; bir LLM'nin **nasıl geliştirildiğini, optimize edildiğini, hizalandığını ve üretim ortamına taşındığını** uçtan uca anlamaktır.

Projeler, klasik NLP yöntemlerinden başlayarak Transformer mimarisi, Fine-Tuning, Quantization, RLHF ve mikroservis tabanlı dağıtıma kadar modern LLM ekosistemini kapsamaktadır.

---

# 🏛️ Öğrenme Yol Haritası

```text
                 Büyük Dil Modelleri (LLM)

                         │
        ┌────────────────┼────────────────┐
        │                │                │
     NLP Temelleri   Transformer      Deployment
        │                │                │
   TF-IDF, Word2Vec  Self-Attention   FastAPI
        │                │            Docker
        └───────────────┬────────────────┘
                        │
                 Fine-Tuning (PEFT)
                        │
                  Quantization
                        │
                    RLHF & DPO
```

---

# 📂 Proje Modülleri

| Modül | Konu | Kazanım |
|:------|:-----|:---------|
| **01-03 NLP Basics** | Metin Ön İşleme | Tokenization, TF-IDF, Word2Vec |
| **04 Transformer** | Transformer Mimarisi | Encoder, Decoder, Multi-Head Attention |
| **05 Self-Attention** | Attention Mekanizması | Query, Key, Value hesaplamaları |
| **06 Autoregression** | Metin Üretimi | Temperature Sampling, Greedy Search |
| **07 Fine-Tuning** | Transfer Learning | PEFT, LoRA, Hugging Face Trainer |
| **08 Quantization** | Model Optimizasyonu | INT8, 4-bit Quantization |
| **09 RLHF & DPO** | Model Hizalama | İnsan tercihleri ile optimizasyon |
| **10 Microservices** | Üretim Ortamı | FastAPI, REST API, Docker |

---

# ⚙️ Mühendislik Pipeline'ı

```text
Ham Metin
     │
     ▼
Metin Ön İşleme
     │
     ▼
Embedding Oluşturma
     │
     ▼
Transformer Modeli
     │
     ▼
Self-Attention
     │
     ▼
Metin Üretimi
     │
     ▼
Fine-Tuning
     │
     ▼
Quantization
     │
     ▼
RLHF / DPO
     │
     ▼
FastAPI Servisi
     │
     ▼
Docker ile Dağıtım
```

---

# 🛠️ Kullanılan Teknolojiler

## 💻 Programlama

- Python

## 🤖 Derin Öğrenme

- PyTorch
- Hugging Face Transformers
- PEFT
- TRL

## 📝 NLP

- NLTK
- SpaCy
- Scikit-Learn

## ⚡ Model Optimizasyonu

- LoRA
- QLoRA
- BitsAndBytes
- Quantization

## 🌐 Backend

- FastAPI
- REST API

## ☁️ DevOps

- Docker

---

# 🧩 Öğrenilen Konular

### 📌 NLP

- Text Cleaning
- Tokenization
- Stopword Removal
- Lemmatization
- TF-IDF
- N-Gram
- Word2Vec

---

### 📌 Transformer

- Positional Encoding
- Multi-Head Attention
- Feed Forward Network
- Residual Connection
- Layer Normalization
- Encoder
- Decoder

---

### 📌 LLM

- Autoregressive Generation
- Temperature Sampling
- Top-K Sampling
- Top-P Sampling
- Greedy Search
- Beam Search

---

### 📌 Fine-Tuning

- Transfer Learning
- PEFT
- LoRA
- QLoRA
- Hugging Face Trainer

---

### 📌 Model Optimizasyonu

- Quantization
- Bellek Optimizasyonu
- Düşük Bitli Model Çalıştırma
- VRAM Optimizasyonu

---

### 📌 Model Hizalama

- RLHF
- DPO
- Reward Modeling
- Preference Learning

---

### 📌 Production AI

- FastAPI
- REST API
- Docker
- API Gateway
- Mikroservis Mimarisi
- Ölçeklenebilir Model Servisi

---

# 🎯 Bu Depodan Neler Öğreneceksiniz?

Bu projeyi tamamladıktan sonra;

✅ Klasik NLP yöntemlerini

✅ Transformer mimarisini

✅ Self-Attention mekanizmasını

✅ Büyük Dil Modellerinin çalışma mantığını

✅ Fine-Tuning süreçlerini

✅ PEFT ve LoRA tekniklerini

✅ Quantization yöntemlerini

✅ RLHF ve DPO ile model hizalamayı

✅ FastAPI ile LLM servisleri geliştirmeyi

✅ Docker ile üretim ortamına dağıtımı öğrenmiş olacaksınız.

---

# 🚀 Kurulum

Projeyi klonlayın

```bash
git clone https://github.com/kullaniciadi/LLM-Engineering.git

cd LLM-Engineering
```

Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

Ardından notebookları sırasıyla çalıştırarak öğrenme yolunu takip edebilirsiniz.

---

# 💡 Mühendislik Yaklaşımı

Bu depo **Model Odaklı** değil, **Sistem Odaklı Yapay Zekâ Mühendisliği** yaklaşımını benimsemektedir.

Odak noktası yalnızca model eğitmek değildir.

Bir LLM'nin;

- Nasıl geliştirildiği,
- Nasıl optimize edildiği,
- Nasıl ince ayar (Fine-Tuning) yapıldığı,
- Nasıl hizalandığı,
- Nasıl API olarak sunulduğu,
- Nasıl ölçeklendirildiği

adım adım ele alınmaktadır.

---

# 📚 Kaynaklar

- Attention Is All You Need
- Hugging Face Transformers
- PyTorch Documentation
- PEFT Documentation
- TRL Documentation
- FastAPI Documentation

---

<div align="center">

### ⭐ Projeyi faydalı bulduysanız yıldız vermeyi unutmayın.

**Geliştiren:** **Şevval Mıkçı**

</div>
