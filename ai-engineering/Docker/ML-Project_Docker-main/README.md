# 🚢 Titanic ML Pipeline

Uçtan uca makine öğrenmesi projesi: **Random Forest · XGBoost · SVM** karşılaştırması ve interaktif Streamlit dashboard'u — tam Docker desteğiyle.

---

## 📁 Proje Yapısı

```
ml-project/
├── src/
│   ├── train.py        # Veri işleme + model eğitimi
│   └── app.py          # Streamlit web arayüzü
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh       # Eğit → Başlat akışı
├── requirements.txt
└── README.md
```

---

## 🚀 Hızlı Başlangıç (Docker Compose — Önerilen)

### Gereksinimler
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (veya Docker Engine + Docker Compose)

### 1. İmajı build et ve başlat

```bash
# Projeyi klonla veya dosyaları indir, sonra:
cd ml-project

docker compose up --build
```

> İlk çalıştırmada:
> 1. Python imajı indirilir (~200 MB)
> 2. Bağımlılıklar yüklenir
> 3. Titanic veri seti otomatik indirilir
> 4. 3 model eğitilir (~1–2 dk)
> 5. Dashboard **http://localhost:8501** adresinde açılır

### 2. Tarayıcıda aç

```
http://localhost:8501
```

### 3. Durdur

```bash
docker compose down          # Modelleri koru
docker compose down -v       # Modelleri de sil (sıfırdan başla)
```

---

## 🐳 Adım Adım Docker Kılavuzu

### Seçenek A — Docker Compose (Önerilen)

```bash
# Build + başlat (arka planda)
docker compose up --build -d

# Logları takip et
docker compose logs -f

# Sadece yeniden build et (kodu değiştirdiysen)
docker compose up --build

# Çalışan container'a gir
docker compose exec ml-app bash

# İçeride sadece eğitimi tekrar çalıştır
docker compose exec ml-app python src/train.py
```

### Seçenek B — Saf Docker Komutları

```bash
# 1. Build
docker build -t titanic-ml .

# 2. Çalıştır (volumes ile)
docker run -d \
  --name titanic-ml-app \
  -p 8501:8501 \
  -v titanic-models:/app/models \
  -v titanic-data:/app/data \
  titanic-ml

# 3. Logları izle
docker logs -f titanic-ml-app

# 4. Durdur & temizle
docker stop titanic-ml-app && docker rm titanic-ml-app
```

---

## 💻 Yerel Geliştirme (Docker Olmadan)

```bash
# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Model eğitimi (data/ ve models/ klasörlerini oluşturur)
python src/train.py

# Dashboard başlat
streamlit run src/app.py
```

---

## 🤖 Modeller

| Model | Parametreler | Özellik |
|-------|-------------|---------|
| **Random Forest** | 200 ağaç, max_depth=8 | Özellik önem skoru |
| **XGBoost** | 200 iter, lr=0.05 | Gradient boosting, hız |
| **SVM** | RBF kernel, C=10 | Küçük veri setlerinde güçlü |

### Özellik Mühendisliği

| Özellik | Açıklama |
|---------|----------|
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | FamilySize == 1 |
| `Title` | İsimden çıkarılan unvan (Mr/Miss/Mrs…) |
| `Sex` | Binary encode |
| `Embarked` | S=0, C=1, Q=2 |

---

## 📊 Dashboard Sayfaları

| Sayfa | İçerik |
|-------|--------|
| **Model Karşılaştırma** | Accuracy, F1, ROC-AUC, Confusion Matrix, CV skorları |
| **Canlı Tahmin** | Yolcu bilgisi gir → model tahmin et |
| **Veri Analizi** | Hayatta kalma dağılımı, yaş histogramı, korelasyon matrisi |

---

## 🔧 Sık Karşılaşılan Sorunlar

**Port çakışması:**
```bash
# docker-compose.yml içinde portu değiştir:
ports:
  - "8502:8501"
```

**Modelleri sıfırla:**
```bash
docker compose down -v
docker compose up --build
```

**İmajı yeniden build et (kod değişti):**
```bash
docker compose up --build --force-recreate
```

---

## 📦 Bağımlılıklar

```
scikit-learn 1.4  |  xgboost 2.0  |  pandas 2.2
numpy 1.26        |  streamlit 1.35  |  plotly 5.22
```

---

## 🗺️ Sonraki Adımlar

- [ ] Kaggle API ile otomatik veri indirme
- [ ] MLflow ile deney takibi
- [ ] FastAPI REST endpoint ekleme
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Kubernetes deployment manifesti
