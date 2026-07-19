# RLHF vs. DPO — Aynı Matematik, İki Farklı Yol

Bu proje, büyük dil modellerini insan tercihleriyle hizalamak (alignment) için
kullanılan iki temel yöntemi — **RLHF** (Reinforcement Learning from Human
Feedback) ve **DPO** (Direct Preference Optimization) — sıfırdan, gerçek
matematikleriyle, tek bir sentetik ortam üzerinde birebir karşılaştıran
çalışan bir demodur.

Teorik/kavramsal anlatım için `docs/RLHF_DPO_Konu_Anlatimi.docx` dosyasına bakın.

## 📁 Proje Yapısı

```
rlhf-dpo-project/
├── src/
│   ├── environment.py     # Sentetik ortam: gizli kalite + Bradley-Terry tercih verisi
│   ├── reward_model.py    # RLHF Adım 1: Ödül modeli (Bradley-Terry MLE)
│   ├── rlhf.py             # RLHF Adım 2: KL-düzenlemeli politika optimizasyonu + kapalı form
│   ├── dpo.py               # DPO: tercih verisinden DOĞRUDAN politika optimizasyonu
│   ├── demo.py               # Ana script: her iki yolu çalıştırır, karşılaştırır, görselleştirir
│   └── test_core.py          # Sayısal gradyan kontrolü + yakınsama testleri
├── outputs/                  # demo.py çalıştırıldığında üretilen grafik ve rapor
├── docs/
│   └── RLHF_DPO_Konu_Anlatimi.docx   # Öğretici doküman
├── requirements.txt
└── README.md
```

## 🧠 Kurgu

Gerçek bir dil modeli eğitmek yerine, matematiği aynı kalacak şekilde
basitleştirilmiş bir "contextual bandit" ortamı kullanılır:

- 5 farklı prompt, her biri için 6 olası cevap.
- Her (prompt, cevap) çiftinin gizli/gerçek bir kalite skoru var (`true_quality`)
  — bu, gerçek RLHF'de asla doğrudan gözlemlenemeyen "insan zihnindeki" tercihtir.
- Bu gizli skordan, **Bradley-Terry modeliyle** gürültülü ikili tercih verisi
  üretilir: `P(i, j'ye tercih edilir) = sigmoid(quality_i − quality_j)`.
- Referans politika `pi_ref`, hizalanmamış (SFT sonrası) modeli temsil eder.

Cevap uzayı küçük ve ayrık olduğu için beklenen değerler **tam olarak**
hesaplanabilir; bu da örnekleme gürültüsü olmadan RLHF'in PPO'da optimize
ettiği aynı amaç fonksiyonunu doğrudan gradyan çıkışıyla optimize etmemizi
sağlar — yani "sonsuz örneklemli PPO" gibi düşünülebilir.

## ⚙️ Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
cd src
python demo.py
```

Testleri çalıştırmak için (DPO gradyanının sayısal olarak doğrulanması dahil):

```bash
cd src
python test_core.py
```

## 📊 demo.py Ne Yapar?

1. **Ortamı kurar**, 250 tercih çifti üretir.
2. **YOL A — RLHF**:
   - Ödül modelini (`r_phi`) Bradley-Terry kaybıyla eğitir.
   - KL-düzenlemeli politika optimizasyonunu gradyan çıkışıyla çalıştırır.
   - Sonucu, kapalı-form optimal çözümle (`pi_ref * exp(r/beta)`) karşılaştırıp
     doğrular.
3. **YOL B — DPO**: Aynı tercih verisinden, ödül modeli hiç eğitmeden,
   politikayı doğrudan DPO kaybıyla optimize eder.
4. **Karşılaştırır**: beklenen gerçek kalite, KL uzaklığı, RLHF'in ödül
   modeli ile DPO'nun "örtük ödülü" arasındaki korelasyon (~0.98).
5. `outputs/rlhf_vs_dpo.png` (6 panelli grafik) ve `outputs/report.txt` üretir.

## 🔑 Temel Çıkarım

RLHF ve DPO, **aynı KL-düzenlemeli hedefi** optimize eder; RLHF bunu iki
aşamada (ödül modeli + RL) yaparken, DPO tek bir denetimli kayıp
fonksiyonuyla yapar. Bu proje, ikisinin de pratikte aynı politikaya
yakınsadığını (RLHF ödülü ile DPO'nun örtük ödülü arasında ~0.98
korelasyon) sayısal olarak gösterir.
