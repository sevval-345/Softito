# Sentiment-Dashboard

Sentiment-Dashboard, metin tabanlı verilerin duygusal tonunu (pozitif, negatif, nötr) belirlemek, verileri görselleştirmek ve kullanıcı etkileşimini analiz etmek için geliştirilmiş kapsamlı bir **React** uygulamasıdır.

## 🧠 Projenin Amacı
Bu uygulama, büyük veri setlerini veya kullanıcı yorumlarını saniyeler içinde işleyerek anlamlı veriye dönüştürür. Kullanıcılar bir metin girdisi sağladığında, arka planda çalışan algoritmalar sayesinde metnin duygusal ağırlığı hesaplanır ve bu sonuçlar dinamik kartlar, istatistik çubukları ve grafiklerle kullanıcıya sunulur.

## 🛠 Kullanılan Teknolojiler

Bu proje, modern web geliştirme standartlarına uygun olarak şu teknolojilerle inşa edilmiştir:

* **React (v18+):** Kullanıcı arayüzünü bileşen (component) tabanlı bir mimariyle yönetmek için kullanılan temel kütüphanedir.
* **Vite:** Geleneksel Webpack kurulumlarına göre çok daha hızlı derleme süresi ve geliştirme deneyimi sunan modern yapılandırma aracıdır.
* **CSS Modules:** Her bileşene özel (`.module.css`) stil dosyaları oluşturarak, stil çakışmalarını önleyen ve CSS kapsamını (scope) modüler hale getiren stil yönetim sistemidir.
* **JavaScript (ES6+):** Uygulama mantığı ve veri işleme süreçleri için kullanılan modern programlama dili.
* **Hooks (useState, useEffect, vb.):** Bileşen durumlarını yönetmek, yan etkileri (API istekleri, veri güncelleme vb.) kontrol etmek için kullanılan React özellikleri.

## 📂 Mimari Yapı

Proje, temiz kod (clean code) prensiplerine göre yapılandırılmıştır:

```text
src/
├── 🧩 components/     # UI: SentimentChart, ResultCard, Chatbot...
├── 🔗 hooks/          # Özel React kancaları ve veri akış yönetimi
├── 🎨 styles/         # Global ve ortak stil dosyaları
├── 🛠 utils/          # LLM servisleri ve yardımcı algoritmalar
└── ⚛️ App.jsx         # Uygulamanın kalbi,

## 📊 Özellikler ve Bileşen İşlevleri

1.  **CommentInput**: Kullanıcının metin girdiği ana form alanı.
2.  **ResultCard**: Analiz sonucunun özetlendiği, görsel olarak vurgulanan bilgi kartı.
3.  **SentimentChart**: Duygu dağılımını (yüzdelik dilimler halinde) gösteren görselleştirme aracı.
4.  **StatsBar**: Uygulamanın anlık genel istatistiklerini (toplam analiz, ortalama skor vb.) gösteren panel.
5.  **HistoryList**: Yapılan geçmiş analizlerin kaydını tutan ve erişim sağlayan liste bileşeni.
6.  **ErrorBanner**: API hataları veya geçersiz giriş durumlarında kullanıcıya geri bildirim veren uyarı sistemi.

## ⚙️ Kurulum ve Başlatma

1.  **Depoyu Klonlayın:**
    `git clone <proje-linkiniz>`

2.  **Dizine Girin:**
    `cd Sentiment-Dashboard`

3.  **Paketleri Yükleyin:**
    `npm install`

4.  **Geliştirme Sunucusunu Başlatın:**
    `npm run dev`
