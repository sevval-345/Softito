# 📊 Veri Mühendisliği Modülleri (Data Engineering)

Büyük veri sistemleri ve ETL/ELT pipeline'larının eğitim amaçlı simülasyonlarını içeren kapsamlı modül koleksiyonu.

---

## 📁 Klasör Yapısı

```
ai-engineering/Data_Engineering/
├── airflow/                          # Apache Airflow ETL Pipeline Simülasyonu
│   └── etl_airflow_pipeline.py      # Satış Verisi ETL Pipeline (DAG, Task, XCom)
│
└── bigdata/                          # Büyük Veri Sistemleri Simülasyonları
    ├── main.py                      # Ana menü ve entegrasyon modülü
    ├── mini_dfs.py                  # HDFS (Dağıtık Dosya Sistemi)
    ├── mini_yarn.py                 # YARN (Kaynak Yöneticisi)
    └── mini_spark.py                # Apache Spark (Veri İşleme Motoru)
```

---

## 🚀 Bölüm 1: Airflow ETL Pipeline

### `etl_airflow_pipeline.py`

**Amaç:** Apache Airflow'un ETL/ELT pipeline yönetim mimarisini Python ile sıfırdan simüle eder.

#### 📋 Temel Yapı

```
EXTRACT → TRANSFORM → LOAD → VALIDATE → NOTIFY
```

#### 🔧 Bileşenler

| Bileşen | Açıklama | Fonksiyon |
|---------|----------|----------|
| **DAG** | Yönlendirilmiş Asiklik Graf | İş akışının konteyneri |
| **Task** | Tek bir iş birimi | DAG içindeki çalıştırılacak görev |
| **Operator** | İşlem tipi | PythonOperator: Python kodu çalıştırır |
| **XCom** | Cross-Communication | Task'lar arası veri paylaşımı |
| **TaskGroup** | Mantıksal gruplama | Görevleri organize etme |
| **Scheduler** | Zamanlamacı | Otomatik çalıştırma zamanı |

#### 📊 Pipeline Aşamaları

1. **EXTRACT (Veri Çekme)**
   - `extract_sales_data()`: Satış verisi çekme (API/DB simülasyonu)
   - `extract_customer_data()`: Müşteri master data çekme
   - Çıktı: Ham veriler XCom'a aktarılır

2. **TRANSFORM (Dönüşüm & Zenginleştirme)**
   - `clean_and_validate_data()`: 
     - Null değer işleme
     - Veri tipi dönüşümü
     - Duplicate kaldırma
   - `enrich_and_aggregate()`:
     - Türetilmiş kolonlar (brüt tutar, indirim, net tutar)
     - Customer segmentleri ile birleştirme
     - Bölge × kategori bazında özet

3. **LOAD (Hedef Sisteme Yazma)**
   - `load_to_data_warehouse()`: SQLite DB'ye yazma
   - Tablolar:
     - `fact_sales`: Satış gerçek tablosu
     - `dim_region_category`: Bölge/kategori boyutu
     - `fact_monthly_trend`: Aylık trend

4. **VALIDATE (Kalite Kontrol)**
   - `run_data_quality_checks()`:
     - Minimum kayıt sayısı kontrolü
     - Negatif tutar kontrolü
     - Null ID kontrolü
     - İskonto aralığı kontrolü
     - Bölge çeşitliliği kontrolü

5. **NOTIFY (Bildirim)**
   - `send_pipeline_notification()`: Slack/E-posta bildirim
   - Metrikler: Toplam ciro, kalite skoru, temizlenen kayıtlar

#### 📤 Çıktılar

```
✅ Pipeline Sonuç Özeti
├─ Başarılı task sayısı
├─ Başarısız task sayısı
├─ Atlanan task sayısı
├─ Toplam çalışma süresi
├─ XCom Değerleri (task arası veri)
└─ Data Warehouse Özeti
   ├─ fact_sales satırları
   ├─ dim_region_category satırları
   ├─ fact_monthly_trend satırları
   └─ pipeline_runs metadata
```

#### 🎯 Kullanım

```bash
python etl_airflow_pipeline.py
```

**Çıktı Örneği:**
```
🚀  DAG ÇALIŞIYOR: etl_satis_pipeline
📅  Execution Date: 2024-01-15 03:00:00
📋  Açıklama: Satış verisi ETL pipeline'ı — günlük çalışır

  📥  Task: extract_sales_data  [EXTRACT]
      ✓  Süre: 0.12s

  ⚙️  Task: clean_and_validate_data  [TRANSFORM]
      ✓  Süre: 0.08s

📊  DAG TAMAMLANDI: etl_satis_pipeline
⏱   Toplam süre  : 1.45 saniye
✅  Başarılı     : 7 task
❌  Başarısız    : 0 task
⏭   Atlandı      : 0 task
```

---

## 🚀 Bölüm 2: Büyük Veri Sistemleri

### `main.py` - Etkileşimli Menü

3 büyük veri bileşenini test etmek için etkileşimli menü sağlar.

```
BÜYÜK VERİ SİSTEMLERİ - MİNİ SİMÜLASYON
==========================================

[1] MINI HDFS      - Dağıtık Dosya Sistemi
[2] MINI YARN      - Küme Kaynak Yöneticisi
[3] MINI SPARK     - Veri İşleme Motoru
[4] TÜMÜNÜ ÇALIŞTIR
[0] ÇIKIŞ
```

---

### `mini_dfs.py` - HDFS Simülasyonu

**Amaç:** Hadoop Distributed File System'i simüle eder.

#### 🏗️ Mimarisi

```
┌─────────────────────────────────────────┐
│        MiniHDFS (Client API)            │
└────────────────┬────────────────────────┘
                 │
         ┌───────▼────────┐
         │   NameNode     │ ← Metadata yönetimi
         │ (Metadata DB)  │
         └────────────────┘
              │         │
        ┌─────▼──┬──────▼────┬──────┐
        │        │           │      │
    DataNode1 DataNode2 DataNode3  ...
    (Block)   (Block)   (Block)
```

#### 🔑 Temel Kavramlar

| Kavram | Açıklama |
|--------|----------|
| **DataNode** | Block'ları fiziksel olarak depolayan sunucu |
| **NameNode** | Dosya sisteminin "beyni", metadata yönetir |
| **Block** | Dosyanın bölündüğü sabit boyutlu parçalar (varsayılan: 128MB) |
| **Replication** | Her block'ün birden fazla kopyası (varsayılan: 3) |
| **Fault Tolerance** | Bir node düşse bile veri erişime açık |

#### 📤 Çıktı Örneği

```
MINI HDFS DEMO
==============

  [NameNode] DataNode kaydedildi: sunucu-1
  [NameNode] DataNode kaydedildi: sunucu-2
  [NameNode] DataNode kaydedildi: sunucu-3

  [NameNode] Dosya depolanıyor: /user/test/dosya.txt
    Block block_0000: 50 bytes -> ['sunucu-1', 'sunucu-2']
    Block block_0001: 50 bytes -> ['sunucu-2', 'sunucu-3']

MINI HDFS STATUS
================
DataNode sayısı: 3
  DataNode(sunucu-1, 2 blocks)
  DataNode(sunucu-2, 2 blocks)
  DataNode(sunucu-3, 1 blocks)
Dosya sayısı: 1
  /user/test/dosya.txt: 2 blocks
```

#### 🎯 Kullanım

```bash
python mini_dfs.py
```

---

### `mini_yarn.py` - YARN Simülasyonu

**Amaç:** Yet Another Resource Negotiator (Kaynak Yöneticisi) simüle eder.

#### 🏗️ Mimarisi

```
┌───────────────────────────────────────────────┐
│      ResourceManager (RM)                     │
│      - Container tahsisi                      │
│      - Scheduling (FIFO / Fair)               │
└────────────┬────────────────┬─────────────────┘
             │                │
        ┌────▼────┐      ┌───▼─────┐
        │NodeMgr-1│      │NodeMgr-2│  ...
        │CPU:4    │      │CPU:8    │
        │RAM:8GB  │      │RAM:16GB │
        │         │      │         │
        │Container│      │Container│
        │ (Task)  │      │ (Task)  │
        └─────────┘      └─────────┘
```

#### 🔑 Temel Kavramlar

| Bileşen | Açıklama |
|---------|----------|
| **ResourceManager** | Kümenin patronu, kaynakları tahsis eder |
| **NodeManager** | Her sunucuda çalışan ajan, container'ları yönetir |
| **ApplicationMaster** | Her uygulamanın kendi yöneticisi |
| **Container** | CPU/RAM birimi, işler burada çalışır |
| **Scheduler** | FIFO veya Fair; kaynakları tahsis stratejisi |

#### 📤 Çıktı Örneği

```
MINI YARN DEMO
==============

  [RM] NodeManager kaydedildi: NodeManager(worker-1, avail=Resource(cpu=4, mem=8192MB))
  [RM] NodeManager kaydedildi: NodeManager(worker-2, avail=Resource(cpu=8, mem=16384MB))

  [RM] Uygulama alındı: app-wordcount (user=user)
  [AM:app-wordcount] Uygulama başlatılıyor...

  [RM] Container tahsis: container_app-wordcount_000 (Resource(cpu=1, mem=512MB)) 
       -> worker-1 (scheduler=fair)
  [AM:app-wordcount] Task 0 çalıştırılıyor @ worker-1...
        [Task] Kelime sayımı tamam: 8 unique kelime

MINI YARN STATUS (scheduler=fair)
==================================
Düğümler: 3
Toplam Kaynak: CPU=14, RAM=28672MB
Kullanılan: CPU=0, RAM=0MB
Uygulamalar: Toplam=1, Çalışan=0
```

#### 🎯 Kullanım

```bash
python mini_yarn.py
```

---

### `mini_spark.py` - Apache Spark Simülasyonu

**Amaç:** Spark'ın RDD, transformations, actions ve DAG kavramlarını simüle eder.

#### 🏗️ Mimarisi

```
Data → RDD → Transformation (lazy) → Transformation (lazy) → Action (eager)
         │                                                         │
         └─────────────── DAG (Lineage) ──────────────────────────┘
```

#### 🔑 Temel Kavramlar

| Kavram | Açıklama | Örnek |
|--------|----------|-------|
| **RDD** | Resilient Distributed Dataset | Partition'lara bölünmüş veri |
| **Partition** | RDD'nin bölündüğü veri parçası | Her partition ayrı node'da işlenir |
| **Transformation** | Lazy işlem, RDD → RDD | map, filter, flatMap, reduceByKey |
| **Action** | Eager işlem, sonuç döndürür | collect, count, reduce, take |
| **DAG** | Directed Acyclic Graph | RDD'ler arasındaki bağımlılıklar |
| **Lineage** | RDD'nin soy ağacı | Kayıp partition'lar yeniden hesaplanabilir |
| **Shuffle** | Veri yeniden dağıtması | reduceByKey sırasında partition'lar arasında |

#### 📊 WordCount Pipeline

```
lines (RDD)
    │
    ├─► flatMap (split) → words (RDD)
    │
    ├─► map (word, 1) → pairs (RDD)
    │
    ├─► reduceByKey (sum) ──[SHUFFLE]──► counts (RDD)
    │
    └─► collect() → Sonuç Array
```

#### 📤 Çıktı Örneği

```
MINI SPARK DEMO - WordCount
===========================

  [SparkContext] Başlatıldı: WordCount, partition=3

  WordCount Pipeline kuruluyor...

  RDD Lineage (Soy Ağacı):
  RDD(reduceByKey(flatMap(text_file)))
    RDD(flatMap(text_file))
      RDD(text_file)

  DAG (Directed Acyclic Graph) - İşlem Akışı
  ===================================================
    └─ RDD(text_file)
    └─ RDD(flatMap(text_file))
    └─ RDD(map(flatMap(text_file)))
    └─ RDD(reduceByKey(map(...)))
  ===================================================
  Actions: collect

  SONUÇ (Kelime Frekansları):
  ════════════════════════════
    bigdata         : 2
    hadoop          : 1
    yarn            : 2
    spark           : 3
    faster          : 1
    ...
  ════════════════════════════
  Toplam 15 farklı kelime
```

#### 🎯 Kullanım

```bash
python mini_spark.py
```



## 📊 Bileşen Karşılaştırması

| Özellik | HDFS | YARN | Spark | Airflow |
|---------|------|------|-------|---------|
| **Amaç** | Veri depolama | Kaynak yönetimi | Veri işleme | Pipeline orchestration |
| **Ölçek** | Terabayt+ | Küme yönetimi | Petabayt+ | Workflow yönetimi |
| **Hız** | Yavaş (disk I/O) | Orta | Hızlı (in-memory) | Bağımlı |
| **Fault Tolerance** | Replication | Container restart | Lineage | Task retry |

---

## 🔍 Gerçek Hayatta Kullanım

```
┌─────────────┐
│   Veri      │
│  Kaynağı    │
│  (API/DB)   │
└──────┬──────┘
       │
       ▼
   ┌───────────┐      (HDFS'ye yazma)
   │  ETL Job  │◄─────────────┐
   └──────┬────┘              │
          │                   │
          ▼                   │
   ┌───────────┐        ┌─────────┐
   │  Airflow  │◄───────┤  HDFS   │
   │  (DAG)    │ (veri  │ (depo)  │
   └──────┬────┘ okuma) └─────────┘
          │
          ▼
   ┌───────────────┐
   │  YARN +       │
   │  Spark Job    │ (MapReduce/Spark)
   └──────┬────────┘
          │
          ▼
   ┌───────────────┐
   │  Data         │
   │  Warehouse    │
   │  (DW/Data    │
   │   Lake)       │
   └───────────────┘
```



**Geliştirici**: Sevval-345  
