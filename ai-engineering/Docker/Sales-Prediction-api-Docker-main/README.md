 📊 Satış Tahmini API - Docker

Python ile geliştirilmiş, makine öğrenmesi tahminleri, web kazıma yetenekleri ve web panosu içeren mikro hizmetler tabanlı bir satış tahmin sistemi. Tüm uygulama, kolay dağıtım için Docker ve Docker Compose kullanılarak konteynerize edilmiştir.

## 🎯 Proje Özeti

Bu proje, birden fazla hizmetiyle tam bir satış tahmin akışı uygular:

- **ML Hizmeti**: Satış tahminleri için makine öğrenmesi modeli
- **Kazıyıcı Hizmeti**: Veri toplamak için web kazıma
- **Web Panosu**: Görselleştirme ve analiz için Flask tabanlı web arayüzü
- **Veritabanı**: Kalıcı veri depolama için PostgreSQL
- **Önbellek**: Önbelleğe alma ve mesaj kuyruğu için Redis

## 🏗️ Mimari

```


┌─────────────────────────────────────────────────┐
│         Web Dashboard (Flask)                   │
│         Port: 5000                              │
└──────────────┬──────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼───┐  ┌───▼────┐
│  ML   │  │Redis │  │Database│
│Service│  │Cache │  │(PG)    │
└───────┘  └──────┘  └────────┘
    │
┌───▼──────┐
│ Scraper  │
└──────────┘
```



## 📋 Hizmetler

### 1. **ML Hizmeti**
Satış tahmini için makine öğrenmesi hizmeti
- Bağımlılıklar: Redis, PostgreSQL
- Tahmin isteklerini işler
- Sonuçları Redis'te önbelleğe alır

### 2. **Kazıyıcı Hizmeti**
Veri toplamak için web kazıma modülü
- Bağımlılıklar: Redis
- Web kaynaklarından veri toplar
- Veri önbelleğe ve veritabanına kaydeder

### 3. **Web Panosu**
Kullanıcı etkileşimi için Flask tabanlı web uygulaması
- Port: **5000**
- Bağımlılıklar: PostgreSQL
- Görselleştirme ve analiz arayüzü sağlar
- Tahminler için kullanıcı dostu arayüz

### 4. **Veritabanı (PostgreSQL)**
Kalıcı veri depolama
- İmaj: `postgres:15`
- Veritabanı: `pricedb`
- Başlatma: `./db/init.sql`
- Kalıcı birim: `postgres_data`

### 5. **Redis Önbelleği**
Bellek içi veri deposu ve mesaj aracı
- İmaj: `redis:alpine`
- Önbelleğe alma ve hizmet iletişimi için kullanılır

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Docker
- Docker Compose

### Kurulum & Başlatma

1. **Depoyu klonlayın**
```bash
git clone https://github.com/sevval-345/Sales-Prediction-api-Docker.git
cd Sales-Prediction-api-Docker
```

2. **Tüm hizmetleri derleyin ve başlatın**
```bash
docker-compose up -d
```

3. **Hizmet durumunu kontrol edin**
```bash
docker-compose ps
```

4. **Web panosuna erişin**
Tarayıcınızı açın ve şuraya gidin:
```
http://localhost:5000
```

### Hizmetleri Durdurma

```bash
docker-compose down
```

Birimler de dahil olmak üzere kaldırmak için:
```bash
docker-compose down -v
```

## 📁 Proje Yapısı

```
Sales-Prediction-api-Docker/
├── docker-compose.yml          # Tüm hizmetler için Compose yapılandırması
├── requirements.txt            # Kök bağımlılıklar
│
├── ml-service/                 # Makine Öğrenmesi Hizmeti
│   ├── Dockerfile
│   └── requirements.txt
│
├── scraper/                    # Web Kazıma Hizmeti
│   ├── Dockerfile
│   └── requirements.txt
│
├── web-dashboard/              # Flask Web Panosu
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
│
└── db/                         # Veritabanı Yapılandırması
    └── init.sql               # PostgreSQL başlatma betiği
```

## 🔧 Yapılandırma

### Ortam Değişkenleri
`docker-compose.yml` içinde veritabanı yapılandırması:
- `POSTGRES_DB`: `pricedb`
- `POSTGRES_PASSWORD`: `password`

Üretim kullanımı için bu değişkenleri compose dosyasında değiştirebilirsiniz.

### Veritabanı Başlatması
SQL başlatma betiklerinizi `./db/init.sql` içine yerleştirin. Bu dosya PostgreSQL başladığında otomatik olarak çalıştırılır.

## 📦 Bağımlılıklar

### Temel Gereksinimler
- `redis` - Önbelleğe alma için Redis Python istemcisi
- `requests` - API çağrıları ve kazıma için HTTP kütüphanesi
- `psycopg2-binary` - Python için PostgreSQL adaptörü

### Hizmetler
- PostgreSQL 15
- Redis (Alpine)
- Python 3.x

## 🐳 Docker Komutları

### Günlükleri Görüntüle
```bash
# Tüm hizmetler
docker-compose logs -f

# Belirli hizmet
docker-compose logs -f web-dashboard
docker-compose logs -f ml-service
docker-compose logs -f scraper
```

### Hizmetleri Yeniden Derle
```bash
docker-compose build
docker-compose up -d
```

### Konteyner İçinde Komut Çalıştır
```bash
# Web panosu
docker-compose exec web-dashboard python app.py

# ML hizmeti
docker-compose exec ml-service python app.py

# Veritabanı
docker-compose exec db psql -U postgres -d pricedb
```

## 💻 Geliştirme

### Yerel Geliştirme Kurulumu (İsteğe Bağlı)
Docker olmadan yerel olarak hizmetleri çalıştırmak istiyorsanız:

1. Python 3.8+ yükleyin
2. PostgreSQL 15 yükleyin
3. Redis yükleyin
4. Sanal ortam oluşturun:
```bash
python -m venv venv
source venv/bin/activate  # Windows'ta: venv\Scripts\activate
```
5. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

## 🔐 Güvenlik Notları

⚠️ **Üretim Kullanımı**: 
- Varsayılan PostgreSQL şifresini değiştirin
- Hassas veriler için ortam değişkenlerini kullanın
- Flask'te uygun kimlik doğrulaması uygulayın
- Docker sırları kullanın
- `.env` dosyaları kullanmayı düşünün (sürüm kontrolünden hariç)

## 📝 API & Kullanım

Ayrıntılı kullanım örnekleri ve API uç noktaları her hizmet dizininde belgelenmiş olmalıdır:
- `ml-service/README.md` - ML tahminleri API
- `scraper/README.md` - Kazıyıcı yapılandırması
- `web-dashboard/README.md` - Pano özellikleri

## 🐛 Sorun Giderme

### Port 5000 Zaten Kullanımda
`docker-compose.yml` içinde port eşlemesini değiştirin:
```yaml
web-dashboard:
  ports:
    - "8000:5000"  # localhost:8000 adresinden erişin
```

### Veritabanı Bağlantı Sorunları
PostgreSQL'in çalışıp çalışmadığını doğrulayın:
```bash
docker-compose logs db
```

### Redis Bağlantı Sorunları
Redis hizmetini kontrol edin:
```bash
docker-compose logs redis
```

### Hizmetler Başlamıyor
İmajları yeniden derleyin:
```bash
docker-compose build --no-cache
docker-compose up -d
```
## 🤝 Katkıda Bulunma

1. Depoyu fork edin
2. Bir özellik dalı oluşturun
3. Değişikliklerinizi commitleyin
4. Dalı push edin
5. Bir Pull Request oluşturun





**Son Güncelleme**: Temmuz 2026





