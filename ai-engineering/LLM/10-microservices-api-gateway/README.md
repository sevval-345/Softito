# Microservices Project — API Gateway + Auth Service + Product Service

Bu proje, `http-proxy-middleware` kullanan bir **API Gateway** üzerinden iki bağımsız mikroservise (`auth-service` ve `product-service`) istek yönlendiren basit bir mimariyi gösterir.

## 📁 Proje Yapısı

```
microservices-project/
├── api-gateway/
│   ├── index.js                  # Gateway ana dosyası, proxy tanımları
│   ├── middlewares/
│   │   ├── requestLogger.js      # Her isteği loglayan middleware
│   │   └── authGuard.js          # x-api-key kontrolü yapan middleware
│   └── package.json
├── auth-service/
│   ├── index.js                  # /login, /register örnek endpoint'leri
│   └── package.json
├── product-service/
│   ├── index.js                  # Ürün listesi örnek endpoint'leri
│   └── package.json
├── package.json                  # Root: concurrently ile hepsini başlatır
└── README.md
```

## ⚙️ Mimari Akış

```
İstemci
  │
  ▼
API Gateway (:3000)
  │  1) requestLogger  -> method, url, timestamp loglar
  │  2) authGuard       -> x-api-key header kontrolü (401 döner ya da geçer)
  │
  ├── /auth/*      ──► Auth Service (:4001)
  └── /products/*  ──► Product Service (:4002)
```

- **Request Logger**: Gateway'e gelen *her* isteğin metodunu, URL'ini ve zaman damgasını konsola yazar. Kimlik doğrulamadan önce çalışır, böylece reddedilen istekler de loglanır.
- **Auth Guard**: `x-api-key` header'ı `secret-123` değilse isteği servislere iletmeden `401 Unauthorized` döner. Geçerliyse istek proxy'e devam eder.
- **Proxy Yönlendirmesi**:
  - `http://localhost:3000/auth/*` → `http://localhost:4001/*`
  - `http://localhost:3000/products/*` → `http://localhost:4002/*`

## 🔑 Varsayılan API Key

```
x-api-key: secret-123
```

Gateway'deki `authGuard.js` içinde `API_KEY` ortam değişkeni ile değiştirilebilir.

## 🚀 Kurulum

Root dizinde, her üç modülün bağımlılıklarını (ve `concurrently`'yi) tek komutla kurun:

```bash
npm install
npm run install:all
```

> `npm install` root `package.json`'daki `concurrently` paketini kurar. `npm run install:all` ise `api-gateway`, `auth-service` ve `product-service` klasörlerindeki `express` / `http-proxy-middleware` bağımlılıklarını kurar.

## ▶️ Çalıştırma

Tüm servisleri **tek komutla** başlatmak için root dizinde:

```bash
npm start
```

Bu komut `concurrently` kullanarak üç servisi aynı anda başlatır:

| Servis | Port | Renk (log) |
|---|---|---|
| api-gateway | 3000 | mavi |
| auth-service | 4001 | yeşil |
| product-service | 4002 | sarı |

Servisleri ayrı ayrı da başlatabilirsiniz:

```bash
npm run start:gateway
npm run start:auth
npm run start:product
```

## 🧪 Test Etme

**Doğru API key ile (başarılı):**

```bash
curl -H "x-api-key: secret-123" http://localhost:3000/auth
curl -H "x-api-key: secret-123" http://localhost:3000/products
```

**Yanlış/eksik API key ile (401 Unauthorized):**

```bash
curl http://localhost:3000/products
# -> {"error":"Unauthorized","message":"Geçersiz veya eksik x-api-key header değeri."}
```

**Login örneği:**

```bash
curl -X POST http://localhost:3000/auth/login \
  -H "x-api-key: secret-123" \
  -H "Content-Type: application/json" \
  -d '{"username":"ahmet","password":"1234"}'
```

**Ürün listesi örneği:**

```bash
curl -H "x-api-key: secret-123" http://localhost:3000/products/
curl -H "x-api-key: secret-123" http://localhost:3000/products/1
```

**Health check (auth guard'a takılmaz):**

```bash
curl http://localhost:3000/health
```

## 📝 Notlar

- Gateway ve servisler arasındaki portlar ortam değişkenleriyle özelleştirilebilir: `PORT`, `AUTH_SERVICE_URL`, `PRODUCT_SERVICE_URL`, `API_KEY`.
- Geliştirme sırasında canlı yeniden başlatma için her modülde `npm run dev` (nodemon) kullanılabilir.
