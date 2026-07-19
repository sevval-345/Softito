import redis
import time
import psycopg2

def get_db_connection():
    while True:
        try:
            return psycopg2.connect(dbname="pricedb", user="postgres", password="password", host="db", port="5432")
        except Exception as e:
            print(f"DB Bekleniyor... Hata: {e}", flush=True)
            time.sleep(2)

db_conn = get_db_connection()
cursor = db_conn.cursor()
r = redis.Redis(host='redis', port=6379, db=0)

print("ML Servisi BAŞLADI", flush=True)
prices = []

while True:
    try:
        data = r.blpop('price_queue', timeout=5)
        if data:
            price = float(data[1].decode('utf-8'))
            prices.append(price)
            print(f"DEBUG: Fiyat alındı: {price}. Liste uzunluğu: {len(prices)}", flush=True)
            
            if len(prices) > 5:
                prices.pop(0)

            # Compute prediction as average of available recent prices (1-5)
            prediction = sum(prices) / len(prices)
            print(f"HESAPLANDI: {prediction}", flush=True)

            # Insert every received price along with current prediction
            try:
                cursor.execute("INSERT INTO price_history (price, prediction) VALUES (%s, %s)", (price, prediction))
                db_conn.commit()
                print(">>> VERİTABANINA YAZILDI", flush=True)
            except Exception as e:
                print(f"DB INSERT HATASI: {e}", flush=True)
                # Try to reconnect on error
                db_conn = get_db_connection()
                cursor = db_conn.cursor()
        
    except Exception as e:
        print(f"!!! KRİTİK DÖNGÜ HATASI: {e}", flush=True)
        # Hata anında bağlantıyı sıfırla
        db_conn = get_db_connection()
        cursor = db_conn.cursor()
        time.sleep(2)