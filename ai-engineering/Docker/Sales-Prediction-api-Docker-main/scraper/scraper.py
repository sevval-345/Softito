import time
import requests
import redis
import sys

# Redis bağlantısı
try:
    # decode_responses=True kullandığınız için veri zaten string olarak gelecek
    redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("Redis'e başarıyla bağlandı!", flush=True)
except redis.ConnectionError:
    print("Redis bağlantı hatası!", flush=True)
    sys.exit(1)

def fetch_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'bitcoin' in data and 'usd' in data['bitcoin']:
                price = data['bitcoin']['usd']
                # Veriyi listeye ekle
                redis_client.lpush('price_queue', price)
                print(f"[{time.strftime('%H:%M:%S')}] Fiyat çekildi ve Redis'e atıldı: {price}", flush=True)
            else:
                print(f"Uyarı: API yanıtında beklenen veri yok. Yanıt: {data}", flush=True)
        else:
            print(f"Hata: API {response.status_code} kodu döndürdü.", flush=True)
            
    except Exception as e:
        print(f"Beklenmedik Hata: {e}", flush=True)

# Döngü
while True:
    fetch_price()
    time.sleep(10) # Nokta hatasını düzelttim