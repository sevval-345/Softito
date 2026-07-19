# 🚀 LLM Engineering & Advanced NLP Pipeline

Bu depo, Büyük Dil Modelleri (LLM) ve Modern NLP süreçlerinin uçtan uca mimarisini anlamaya odaklanan, akademik teoriden üretim odaklı (production-ready) uygulamalara kadar uzanan kapsamlı bir mühendislik çalışmasıdır.

## 🧠 Klasör İçeriği ve Müfredat

### 1. Temel NLP & Model Mimarileri
Doğal dil işleme dünyasına giriş ve Transformer tabanlı sistemlerin matematiksel altyapısı:

*   **`01-Metin_Ön_İşleme.ipynb`**: Tokenization, lemmatization ve vektörize etme süreçleri.
*   **`02-Gelişmiş_Metin_Temsili.ipynb`**: Word Embeddings (Word2Vec, FastText) ve bağlamsal temsil.
*   **`03-Duygu_analizi.ipynb`**: Derin öğrenme ile metin sınıflandırma uygulamaları.
*   **`04-Transformer.py`**: Transformer mimarisinin (Encoder-Decoder) modüler kodlanması.
*   **`05-self_attention.ipynb`**: Attention mekanizmasının matris işlemleriyle detaylı analizi.
*   **`06-sıcaklık_ve_otoregresyon.ipynb`**: Modelin tahmin üretme süreçleri, temperature ve sampling stratejileri.

### 2. İleri Seviye LLM Operasyonları
Modellerin eğitilmesi, ince ayar yapılması ve optimize edilmesi süreçleri:

*   **`07-Fine_Tuning.ipynb`**: Hugging Face ekosistemi ile transfer öğrenme ve model adaptasyonu.
*   **`08-quantization-project/`**: Model sıkıştırma teknikleri ile çıkarım (inference) hızlandırma.
*   **`09-rlhf-dpo-project/`**: İnsan geri bildirimi (RLHF) ve Direct Preference Optimization (DPO) ile model hizalama.
*   **`10-microservices-api-gateway/`**: LLM tabanlı çözümlerin ölçeklenebilir mikroservis mimarileri ile servise açılması.

---

## 🛠 Teknik Yığın (Tech Stack)

*   **Core Frameworks**: PyTorch, Hugging Face (Transformers, PEFT, TRL)
*   **Optimization**: BitsAndBytes (Quantization), LoRA/QLoRA
*   **Deployment**: FastAPI, Docker, Microservices Architecture
*   **NLP Tools**: NLTK, Spacy, Scikit-Learn

---

## 🏗 Mimari Yaklaşım
Bu çalışmalar, **"Model Centric"** yaklaşımdan **"System Centric"** yaklaşıma geçişi hedefler. Sadece bir model eğitmek değil, eğitilen modelin nasıl optimize edileceği, hizalanacağı (alignment) ve kurumsal bir API çatısı altında nasıl sunulacağı süreçlerini kapsar.

> **Mühendislik Notu:** Bu projeler, özellikle LLM'lerin kaynak yönetimi (Quantization) ve insan değerleriyle hizalanması (DPO/RLHF) konularında güncel endüstri standartlarını takip eder.

---

## 🚀 Başlarken

Her klasörün içerisinde o bölüme özel `README.md` ve `requirements.txt` dosyaları bulunmaktadır. Çalıştırmaya başlamadan önce:

```bash
pip install -r requirements.txt
