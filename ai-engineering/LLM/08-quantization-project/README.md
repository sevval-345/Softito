# Kuantizasyon (Quantization) Demo Projesi

Bu proje, derin öğrenme modellerinde yaygın kullanılan **kuantizasyon** tekniğini
sıfırdan (NumPy ile, framework bağımsız) uygulayan çalışan bir demodur.

Teorik anlatım için `docs/Kuantizasyon_Konu_Anlatimi.docx` dosyasına bakın.

## 📁 Proje Yapısı

```
quantization-project/
├── src/
│   ├── quantizer.py         # Simetrik / asimetrik quantize-dequantize matematiği
│   ├── model.py              # Basit bir NumPy MLP (etkiyi göstermek için)
│   ├── demo.py                # Ana script: çalıştır, ölç, görselleştir
│   └── test_quantizer.py      # Doğruluk testleri
├── outputs/                   # demo.py çalıştırıldığında üretilen grafik ve raporlar
├── docs/
│   └── Kuantizasyon_Konu_Anlatimi.docx   # Öğretici doküman
├── requirements.txt
└── README.md
```

## ⚙️ Kurulum

```bash
pip install -r requirements.txt
```

## ▶️ Çalıştırma

```bash
cd src
python demo.py
```

Bu komut:
1. 256x256'lık rastgele bir ağırlık tensörünü hem **simetrik** hem **asimetrik**
   INT8 kuantizasyon ile kuantize eder ve hatayı (MSE, MAE, SQNR) ölçer.
2. Basit bir 2 katmanlı MLP modelinde, ağırlıklar kuantize edildiğinde
   çıktının ne kadar saptığını gösterir (fake-quantization simülasyonu).
3. `outputs/` klasörüne şu dosyaları yazar:
   - `quantization_histograms.png` — orijinal vs. kuantize edilmiş ağırlık dağılımları
   - `bitwidth_tradeoff.png` — bit genişliği arttıkça hata/sıkıştırma dengesi
   - `report.txt` — metin özeti

Testleri çalıştırmak için:

```bash
cd src
python test_quantizer.py
```

## 🧠 Ne Öğretiyor?

- `q = round(x / scale) + zero_point` formülüyü gerçek kodda uygulama
- Simetrik ve asimetrik kuantizasyon arasındaki fark
- `scale` ve `zero_point` parametrelerinin nasıl hesaplandığı
- Kuantizasyonun bir modelin çıktısını ne kadar (ve neden az) değiştirdiği
- Bit genişliği düştükçe hata artışı / bellek tasarrufu ilişkisi (2, 4, 6, 8, 16 bit karşılaştırması)

Kavramsal arka plan, formüllerin türetilişi, INT8 aralığı, PTQ vs QAT gibi
konular için `docs/` klasöründeki Word dosyasını inceleyin.
