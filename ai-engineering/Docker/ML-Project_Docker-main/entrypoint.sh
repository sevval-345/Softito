#!/bin/bash
set -e

echo "======================================================"
echo "  🚢 Titanic ML Pipeline başlatılıyor..."
echo "======================================================"

# Modeller yoksa eğit
if [ ! -f "/app/models/metadata.json" ]; then
  echo ""
  echo "📦 Modeller bulunamadı — eğitim başlıyor..."
  python src/train.py
  echo "✅ Eğitim tamamlandı."
else
  echo "✅ Modeller zaten mevcut, eğitim atlanıyor."
fi

echo ""
echo "🌐 Streamlit arayüzü başlatılıyor → http://localhost:8501"
echo ""

exec streamlit run src/app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
