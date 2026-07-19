# 🤖 AI Engineering Modülü

> Yapay zeka mühendisliğinin temel konularını kapsamlı şekilde ele alan eğitim ve uygulama projelerinin bulunduğu modüldür. Makine öğrenmesi, derin öğrenme, doğal dil işleme, veri mühendisliği ve deployment konuları içeren kapsamlı bir portföy.

---

## 📑 İçerik Tablosu

- [🎯 Genel Bakış](#-genel-bakış)
- [🏗️ Modül Yapısı](#️-modül-yapısı)
- [⚙️ Teknik Altyapı](#️-teknik-altyapı)
- [📚 Ana Bölümler](#-ana-bölümler)
- [🛠️ Kullanılan Teknolojiler](#️-kullanılan-teknolojiler)
- [🚀 Başlangıç Rehberi](#-başlangıç-rehberi)
- [💡 Proje Örnekleri](#-proje-örnekleri)
- [🌐 Sistem Mimarisi](#-sistem-mimarisi)

---

## 🎯 Genel Bakış

Bu modül, **yapay zeka ve makine öğrenmesi** alanında pratik deneyim kazanmak isteyen mühendisler için kapsamlı bir öğrenme ortamı sunmaktadır. Temel konseptlerden başlayarak, üretime hazır uygulamalara kadar her seviyede içerik bulunmaktadır.

### 👥 Hedef Kullanıcılar

- 🔬 Makine öğrenmesi mühendisleri
- 📊 Veri bilimcileri
- 🧠 Yapay zeka araştırmacıları
- 💻 Yazılım mühendisleri (AI/ML geçişi yapmak isteyenler)

### 📊 Seviyeleri
```
🟢 Başlangıç → 🟡 Orta → 🔴 İleri
```

---

## 🏗️ Modül Yapısı

```
ai-engineering/
│
├── 📦 Data_Engineering/              # Veri işleme ve ETL pipeline'ları
│   ├── airflow/                      # Apache Airflow ile iş akışı yönetimi
│   │   └── etl_airflow_pipeline.py
│   └── bigdata/                      # Büyük veri teknolojileri
│       ├── mini_dfs.py               # Dağıtık dosya sistemi simulatörü
│       ├── mini_spark.py             # Apache Spark benzeri MapReduce motoru
│       ├── mini_yarn.py              # YARN benzeri kaynak yöneticisi
│       └── main.py
│
├── 🤖 Machine_Learning/              # Klasik makine öğrenmesi algoritmaları
│   ├── classification/               # Sınıflandırma problemleri
│   │   ├── decision_tree.ipynb
│   │   ├── knn_vs_naive_bayes.ipynb
│   │   ├── adaboost_xgboost_karsilastirma.ipynb
│   │   ├── svm_analiz.ipynb
│   │   ├── temel_ann.ipynb
│   │   ├── derin_ogrenme_giris.ipynb
│   │   ├── face_detection.ipynb
│   │   ├── habersentiment.ipynb
│   │   ├── CNN.ipynb
│   │   └── Cats & Dogs Classification.ipynb
│   │
│   ├── regression/                   # Regresyon analizi
│   │   ├── regresyon_analizi.ipynb
│   │   ├── logistic_regression.ipynb
│   │   └── polinomsal_regresyon_vs_karar_agaci.ipynb
│   │
│   └── unsupervised/                 # Denetimsiz öğrenme
│       ├── unsupervised_algorithms.ipynb
│       ├── simple_rnn.ipynb
│       └── temel_python.ipynb
│
├── 🎨 Deep_Learning_and_CV/          # Derin öğrenme ve bilgisayar görme
│   ├── CNN/                          # Konvolüsyonal sinir ağları
│   │   ├── cnn1.ipynb                # Temel CNN mimarisi
│   │   ├── cnn2.ipynb                # İleri CNN örnekleri
│   │   ├── Mask_R_CNN.ipynb          # Nesne algılama ve segmentasyon
│   │   ├── U_Net.ipynb               # Görüntü segmentasyonu
│   │   ├── yolo.ipynb                # Gerçek zamanlı nesne algılama
│   │   └── README.md
│   │
│   └── Image_Processing/             # Görüntü işleme ve ileri teknikler
│       ├── image_processing_basics.py
│       ├── plaka_pipeline.py
│       ├── Image_processing2.ipynb
│       └── README.md
│
├── 💬 NLP/                           # Doğal dil işleme
│   ├── 01-NLP_Giriş.ipynb
│   ├── 02-tf_idf.ipynb
│   ├── 03-word-vectors.ipynb
│   ├── 04-word_embeddings.py
│   ├── 05-rnn.ipynb
│   ├── 06-LSTM.ipynb
│   ├── 07-attention_mechanisms.ipynb
│   ├── 08-BoW_ve_N_gramlar.ipynb
│   └── 09-bow-tf-df-embedding.ipynb
│
├── 🚀 LLM/                           # Büyük dil modelleri
│   ├── 01-Metin_Ön_İşleme.ipynb
│   ├── 02-Gelişmiş_Metin_Temsili.ipynb
│   ├── 03-Duygu_analizi.ipynb
│   ├── 04-Transformer.py
│   ├── 05-self_attention.ipynb
│   ├── 06-sıcaklık_ve_otoregresyon.ipynb
│   ├── 07-Fine_Tuning.ipynb
│   ├── 08-quantization-project/
│   ├── 09-rlhf-dpo-project/
│   └── 10-microservices-api-gateway/
│
├── 🐳 Docker/                        # Konteyner ve deployment
│   ├── ML-Project_Docker-main/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   ├── entrypoint.sh
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── src/
│   │
│   └── Sales-Prediction-api-Docker-main/
│       ├── docker-compose.yml
│       ├── requirements.txt
│       ├── README.md
│       ├── ml-service/
│       ├── web-dashboard/
│       ├── scraper/
│       └── db/
│
├── 📊 Sentiment Dashboard(Bireysel_Proje)/  # Duygusal analiz dashboard'u
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── README.md
│   └── src/
│
└── README.md                         # 📋 Bu dosya
```

---

## ⚙️ Teknik Altyapı

### 🔬 Kullanılan Başlıca Algoritma ve Mimariler

| Kategori | Algoritmalar/Mimariler |
|----------|---|
| **🔷 Sınıflandırma** | Decision Tree, KNN, Naive Bayes, SVM, AdaBoost, XGBoost, ANN, CNN |
| **📈 Regresyon** | Linear Regression, Polynomial Regression, Logistic Regression |
| **🔄 Denetimsiz Öğrenme** | Clustering (K-Means), Dimensionality Reduction |
| **🧠 Derin Öğrenme** | CNN, RNN, LSTM, Transformer, U-Net, Mask R-CNN, YOLO |
| **💬 NLP** | TF-IDF, Word2Vec, LSTM, Attention, Transformer, BERT |
| **🤖 LLM** | Fine-tuning, RLHF, DPO, Quantization, Temperature Sampling |
| **🔀 Veri Mühendisliği** | Apache Airflow, Apache Spark, YARN, DFS, ETL Pipelines |
| **🚀 Deployment** | Docker, Docker Compose, Microservices, API Gateway |

---

## 📚 Ana Bölümler

### 1️⃣ Veri Mühendisliği (Data_Engineering)

#### Apache Airflow ile İş Akışı Yönetimi
- **📄 Dosya:** `etl_airflow_pipeline.py`
- **🎯 Amaç:** Veri işleme iş akışlarını planlama ve yönetme
- **📌 İçerik:**
  - ✓ DAG (Directed Acyclic Graph) tanımı
  - ✓ Task bağımlılıkları
  - ✓ Hata yönetimi
  - ✓ Logging ve monitoring

#### Büyük Veri Teknolojileri
- **mini_dfs.py:** Dağıtık dosya sisteminin temel işlevi
  - Dosya parçalama ve dağıtım
  - Replikasyon mekanizması
  - Veri bütünlüğü kontrol

- **mini_spark.py:** MapReduce motoru
  - Map ve Reduce operasyonları
  - Distributed computing
  - Data aggregation

- **mini_yarn.py:** Kaynak yöneticisi
  - Job scheduling
  - Resource allocation
  - Task coordination

### 2️⃣ Makine Öğrenmesi (Machine_Learning)

#### 🎯 Sınıflandırma (Classification)
- **Temel Algoritmalar:** Decision Tree, KNN, Naive Bayes, SVM
- **İleri Teknikler:** AdaBoost, XGBoost, Ensemble Methods
- **Derin Öğrenme:** ANN, CNN
- **Uygulamalar:** Yüz tanıma, haber duygusal analizi, hayvan sınıflandırması

**Örnek Proje:** Kedi vs Köpek Sınıflandırması
- Custom CNN mimarisi vs VGG16 karşılaştırması
- Transfer Learning uygulaması
- Model evaluasyonu ve optimizasyon

#### 📊 Regresyon (Regression)
- **Linear Regression:** Temel regresyon analizi
- **Polynomial Regression:** Non-lineer ilişki modellemesi
- **Logistic Regression:** İkilik sınıflandırma
- **Karşılaştırmalı Analizler:** Algoritma performans değerlendirmesi

#### 🔄 Denetimsiz Öğrenme (Unsupervised)
- Clustering algoritmaları
- Dimensionality reduction teknikleri
- Python programlama temelleri
- RNN temel uygulamaları

### 3️⃣ Derin Öğrenme ve Bilgisayar Görme

#### 🔷 Konvolüsyonal Sinir Ağları (CNN)
- **Temel CNN:** Katman-katman mimari
- **U-Net:** Görüntü segmentasyonu
- **Mask R-CNN:** Nesne algılama ve instance segmentasyonu
- **YOLO:** Gerçek zamanlı nesne algılama

#### 🖼️ Görüntü İşleme (Image Processing)
- **image_processing_basics.py:** OpenCV ile temel işlemler
  - Görüntü okuma/yazma
  - Filtreleme ve kenar algılama
  - Renk uzayı dönüşümleri
  - Morfolojik işlemler

- **plaka_pipeline.py:** Plaka tanıma sistemi
  - Görüntü ön işleme
  - Plaka bölgesi tespiti
  - Karakter segmentasyonu
  - OCR uygulaması

### 4️⃣ Doğal Dil İşleme (NLP)

#### 📝 Metin Temsili Yöntemleri
- **TF-IDF:** Terim frekansı-ters belge frekansı
- **Word2Vec:** Word embeddings
- **Bag of Words (BoW):** Basit metin temsili
- **N-Gramlar:** Kelime sekansı analizi

#### 🔄 Sıralı Modeller
- **RNN:** Temel tekrarlayan sinir ağları
- **LSTM:** Long Short-Term Memory
- **GRU:** Gated Recurrent Unit
- **Attention Mechanisms:** Mekanizması odaklanmak

#### 🎯 Word Embeddings
- Otomatik word pool oluşturma
- Semantic kategorize etme
- Corpus oluşturma ve test
- Embedding boyut optimizasyonu

### 5️⃣ Büyük Dil Modelleri (LLM)

#### 📌 Metin Ön İşleme ve Temsili
- Türkçe karakter işleme
- Tokenizasyon teknikleri
- Normalizasyon ve temizleme

#### 🤖 Transformer Mimarisi
- **04-Transformer.py:** Sıfırdan Transformer implementasyonu (NumPy)
  - Embedding ve positional encoding
  - Self-attention mekanizması
  - Multi-head attention
  - Feed-forward networks
  - Residual bağlantılar
  - Layer normalization
  - Classification layer

- **05-self_attention.ipynb:** Detaylı self-attention analizi
  - Query, Key, Value hesaplamaları
  - Attention weights
  - Visualization ve interpretability

#### 🎛️ Model İyileştirme Teknikleri
- **Fine-Tuning:** Önceden eğitilmiş modelleri özelleştirme
- **Temperature Sampling:** Otoregressif jenerasyon kontrolü
- **Quantization:** Model sıkıştırma ve hızlandırma
- **RLHF:** İnsanlar tarafından tercih edilen cevapları öğrenme
- **DPO:** Doğrudan tercih optimizasyonu

#### 😊 Duygusal Analiz (Sentiment Analysis)
- Metin sınıflandırması
- Model değerlendirmesi
- Tahmin çıktısı yorumlama

#### 🏗️ Mikroservis Mimarisi
- API Gateway pattern
- Auth Service implementasyonu
- Product Service yapısı
- Service-to-service iletişim

### 6️⃣ Docker ve Konteynerleştirme

#### 🐳 ML Projesi Docker ile
- **Dockerfile:** Python ortamı ve bağımlılık kurulumu
- **docker-compose.yml:** Multi-container orchestration
- **entrypoint.sh:** Container startup script
- Model servisleme ve API deployment

#### 📊 Satış Tahmin Sistemi
- **Mimarisi:**
  - ML Service: Model ve tahmin motor
  - Web Dashboard: Sonuç görselleştirmesi
  - Scraper: Veri toplama
  - Database: Veri depolama

---
# 🛠️ Kullanılan Teknolojiler

## 💻 Programlama Dilleri

<p>
  <img src="https://skillicons.dev/icons?i=python,js,bash" />
  <img src="https://img.shields.io/badge/SQL-336791?style=for-the-badge&logo=postgresql&logoColor=white"/>
</p>


---

## 🧠 Makine Öğrenmesi

<p>
  <img src="https://skillicons.dev/icons?i=tensorflow,pytorch" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
</p>


---

## 📚 Doğal Dil İşleme (NLP)

<p>
  <img src="https://img.shields.io/badge/NLTK-4CAF50?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/BERT-GPT-blueviolet?style=for-the-badge"/>
</p>

---

## 🔀 Veri İşleme

<p>
  <img src="https://skillicons.dev/icons?i=postgres" />
  <img src="https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
</p>


---

## 🖼️ Görüntü İşleme

<p>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/scikit--image-F7931E?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
</p>

---

## 🚀 Backend & DevOps

<p>
  <img src="https://skillicons.dev/icons?i=docker,fastapi,flask,nginx" />
  <img src="https://img.shields.io/badge/Docker%20Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
</p>



---

## 🎨 Frontend

<p>
  <img src="https://skillicons.dev/icons?i=react,vite,css" />
</p>


---

## 🗄️ Veritabanları

<p>
  <img src="https://skillicons.dev/icons?i=postgres,mongodb,sqlite" />
</p>


---

## 🚀 Başlangıç Rehberi

### 📋 Sistem Gereksinimleri

```bash
# Python sürümü
python >= 3.8

# Temel paketler
pip install numpy pandas scikit-learn matplotlib seaborn

# Derin öğrenme
pip install tensorflow keras torch

# NLP
pip install nltk spacy transformers

# Görüntü işleme
pip install opencv-python pillow scikit-image

# Veri mühendisliği
pip install apache-airflow pyspark

# Web framework'leri
pip install flask fastapi uvicorn

# Frontend
node >= 14.0
npm >= 6.0
```

### ⚡ Temel Kurulum Adımları

#### 1️⃣ Repository'yi Klonla
```bash
git clone https://github.com/sevval-345/Softito.git
cd Softito/ai-engineering
```

#### 2️⃣ Virtual Environment Oluştur
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

#### 3️⃣ Bağımlılıkları Yükle
```bash
# İhtiyaca göre gerekli paketleri yükle
pip install -r requirements.txt
```

#### 4️⃣ Jupyter Notebook Başlat
```bash
jupyter notebook
```

### 💡 Örnek Projelerle Başlangıç

#### 📝 NLP İle Metin Analizi
```bash
cd NLP/
jupyter notebook 01-NLP_Giriş.ipynb
```

#### 🤖 Transformer Modeli Anlamak
```bash
cd LLM/
python 04-Transformer.py
```

#### 🐱 Görüntü Sınıflandırması
```bash
cd Machine_Learning/classification/
jupyter notebook "Cats & Dogs Classification.ipynb"
```

---

## 💡 Proje Örnekleri

### 1️⃣ Sentiment Dashboard (Bireysel Proje)

**📝 Açıklama:** React tabanlı, metin duygusal analizini gerçek zamanlı yapan web uygulaması.

**✨ Özellikler:**
- ✓ Metin girdisi al ve duygusal analiz yapma
- ✓ Sonuçları görsel kartlar halinde gösterme
- ✓ Duygu dağılımı grafiği
- ✓ Geçmiş analizler kütüphanesi
- ✓ İstatistik paneli

**🛠️ Teknoloji Stack:**
- React 18+ (UI)
- Vite (build tool)
- CSS Modules (styling)
- LLM integration (backend)

**🎬 Başlatma:**
```bash
cd Sentiment\ Dashboard\(Bireysel_Proje\)/
npm install
npm run dev
```

### 2️⃣ Satış Tahmin Sistemi (Docker)

**📊 Açıklama:** Satışları tahmin eden ML modelini Docker konteynerinde çalıştıran sistem.

**🏗️ Mimarisi:**
- **ML Service:** Model ve tahmin motor
- **Web Dashboard:** Sonuç gösterimi
- **Scraper:** Veri toplama
- **Database:** Veri saklama

**🎬 Başlatma:**
```bash
cd Docker/Sales-Prediction-api-Docker-main/
docker-compose up -d
# http://localhost:3000 adresinde erişin
```

### 3️⃣ Transformer Mimarisi (Sıfırdan Uygulama)

**🧠 Açıklama:** NumPy kullanarak Transformer mimarisinin temel bileşenlerini uygulayan eğitim projesi.

**📚 Öğrenilen Kavramlar:**
- Tokenization ve Embedding
- Self-Attention mekanizması
- Multi-Head Attention
- Feed-Forward Networks
- Residual bağlantıları
- Layer Normalization
- Sınıflandırma katmanı

**▶️ Çalıştırma:**
```bash
cd LLM/
python 04-Transformer.py
# Çıktı transformer_output.txt dosyasına kaydedilir
```

### 4️⃣ Plaka Tanıma Sistemi

**🎯 Açıklama:** Görüntü işleme ve derin öğrenme kullanarak plaka tanıma.

**🔧 Bileşenler:**
- Görüntü ön işleme (normalize, threshold)
- Bölge tespiti (contour detection)
- Karakter segmentasyonu
- OCR (Optical Character Recognition)

**▶️ Çalıştırma:**
```bash
cd Deep_Learning_and_CV/Image_Processing/
python plaka_pipeline.py --input image.jpg
```

---

## 🌐 Sistem Mimarisi

### 📊 Genel Veri Akışı

```
┌─────────────────────────────────────────────────────┐
│              🔵 Veri Kaynakları                      │
│   (Web Scraping, API, CSV, Veritabanı)             │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│    🟠 Veri Mühendisliği (Data Engineering)          │
│  ┌─────────────┐  ┌──────────┐  ┌────────────┐    │
│  │   Airflow   │  │  Spark   │  │  Mini DFS  │    │
│  │ Orchestr.   │  │ Process. │  │ Storage    │    │
│  └─────────────┘  └──────────┘  └────────────┘    │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│      🟡 Makine Öğrenmesi Pipeline                   │
│  ┌────────────────────────────────────────────┐    │
│  │ Feature Eng. | Model Train | Validation   │    │
│  └────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────┐    │
│  │ Tree, SVM, Linear/Logistic Regression     │    │
│  └────────────────────────────────────────────┘    │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│    🟢 Derin Öğrenme & Özel Görevler                 │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐      │
│  │Computer  │  │  NLP &   │  │  LLM &      │      │
│  │Vision    │  │Transform │  │  Fine Tune  │      │
│  │CNN, YOLO │  │LSTM,BERT │  │RLHF, DPO   │      │
│  └──────────┘  └──────────┘  └─────────────┘      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│         🔴 Model Deployment                         │
│  ┌─────────────────────────────────────────────┐   │
│  │ Docker | Docker Compose | Microservices    │   │
│  │ API Gateway | Load Bal. | Monitoring       │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│      🟣 Kullanıcı Arayüzü & API                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ React Dashboard | REST API | WebSocket      │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 🧩 Bileşen İntegrasyonu

```
┌──────────────────┐
│  📊 Data Pipeline   │ ─── Apache Airflow/Spark
├──────────────────┤
│  🏪 Feature Store   │ ─── Pandas/NumPy
├──────────────────┤
│  📦 Model Registry  │ ─── MLflow/Custom
├──────────────────┤
│  🧠 Model Training  │ ─── TensorFlow/PyTorch
├──────────────────┤
│  🚀 Model Serving   │ ─── Flask/FastAPI
├──────────────────┤
│  🎨 Frontend        │ ─── React/Vue
├──────────────────┤
│  ☁️ Infrastructure  │ ─── Docker/Kubernetes
└──────────────────┘
```

---

## 📖 Ana Algoritmalar ve Metodoloji

### 👁️ Denetimlü Öğrenme (Supervised Learning)

**🔷 Sınıflandırma:**
- Decision Tree: Ağaç yapısı ile karar verme
- K-Nearest Neighbors: Benzerlik temelli sınıflandırma
- Support Vector Machine: Maksimum marjin sınıflandırıcısı
- Naive Bayes: Olasılıksal sınıflandırma
- Ensemble Methods: AdaBoost, XGBoost

**📈 Regresyon:**
- Linear Regression: Doğrusal ilişki modelleme
- Polynomial Regression: N-derece polinom uydurmak
- Logistic Regression: İkilik sınıflandırma
- Ridge/Lasso: Düzenleme ile regresyon

### 🔄 Denetimsiz Öğrenme (Unsupervised Learning)
- K-Means Clustering: Merkez temelli kümeleme
- Hierarchical Clustering: Hiyerarşik yapı
- Dimensionality Reduction: PCA, t-SNE
- Anomaly Detection: İstatistiksel metod

### 🧠 Derin Öğrenme (Deep Learning)

**🏗️ Mimariler:**
- Feedforward Neural Network (FNN)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Transformer
- Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)

### 🔄 Transfer Learning
- ImageNet önceden eğitilmiş modeller
- BERT, GPT gibi dil modelleri
- Fine-tuning ve adaptation teknikleri

---

## 📊 Performans Metrikleri

### 🔷 Sınıflandırma Metrikleri
```
✓ Accuracy: (TP + TN) / (TP + TN + FP + FN)
✓ Precision: TP / (TP + FP)
✓ Recall: TP / (TP + FN)
✓ F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
✓ ROC-AUC: ROC eğrisi altında kalan alan
```

### 📈 Regresyon Metrikleri
```
✓ MAE: Ortalama mutlak hata
✓ MSE: Ortalama kare hata
✓ RMSE: Ortalama kare hatanın karekökü
✓ R²: Belirleme katsayısı
```

### 🎯 Model Değerlendirmesi
- Cross-validation
- Hyperparameter tuning
- Learning curves analiz
- Confusion matrix

---

## 📦 Bağımlılıklar

### 🐍 Python Paketleri
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
tensorflow>=2.8.0
torch>=1.9.0
transformers>=4.20.0
opencv-python>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
apache-airflow>=2.3.0
pyspark>=3.2.0
```

### 🖥️ Sistem Gereksinimleri
- **CPU:** İşlemci (Intel i5/i7 veya eşdeğer)
- **RAM:** Minimum 8GB, önerilen 16GB+
- **GPU:** NVIDIA CUDA destekli (opsiyonel, önerilen)
- **Disk:** 20GB+ boş alan
- **OS:** Linux, macOS, Windows

---

## 🔗 Kaynaklar ve Referanslar

### 📚 Resmi Dokümantasyon
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Apache Airflow Documentation](https://airflow.apache.org/)
- [Docker Documentation](https://docs.docker.com/)

### 📜 Araştırma Makaleleri
- "Attention is All You Need" - Transformer mimarisi
- "ImageNet Classification with Deep CNNs" - ResNet, VGG
- "An Empirical Evaluation of Generic Convolutional Networks" - CNN karşılaştırması

### 🎓 Öğrenme Kaynakları
- Fast.ai Practical Deep Learning
- Andrew Ng - Machine Learning Specialization
- Yandex - Advanced Machine Learning Specialization

---

## 🤝 Katkı Rehberi

### 📝 Yeni Proje Ekleme Adımları

1. Repository'yi fork et
2. Feature branch oluştur (`git checkout -b feature/yeni-proje`)
3. Değişiklikleri commit et (`git commit -am 'Yeni proje eklendi'`)
4. Branch'i push et (`git push origin feature/yeni-proje`)
5. Pull Request oluştur

### 📋 Kod Standartları

- **Python:** PEP 8 standardına uyun
- **Docstring:** NumPy docstring formatı
- **Type Hints:** Python 3.8+ type annotations
- **Testing:** unittest veya pytest kullanın
- **Documentation:** Markdown formatında README

---

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasını kontrol edin.

---

## 📧 İletişim ve Destek

**👤 Proje Sahibi:** sevval-345

**❓ Sorular ve Öneriler:**
- 🐛 GitHub Issues üzerinden bug raporlayın
- 💬 Discussions kısmında soru sorun
- 🔄 Pull Request ile katkıda bulunun

---

## 📌 Notu

Bu modül sürekli geliştirilmekte ve güncellenmektedir. En son değişiklikler için `CHANGELOG.md` dosyasını kontrol edin.

**⏰ Son Güncelleme:** 2024

---

**📍 Sürüm:** 1.0.0  
**🟢 Durum:** Aktif Geliştirme
