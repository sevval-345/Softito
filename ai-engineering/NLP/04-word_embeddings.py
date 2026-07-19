import numpy as np
import re
from collections import Counter
from typing import Dict, List

np.random.seed(42)

# Uzun metin - bu metinden word_pool otomatik olarak çıkartılacak
uzun_metin = """Küresel ekosistemde yaşanan hızlı dönüşüm, insanlığın geleceğini yeniden şekillendirirken farklı disiplinleri de kaçınılmaz olarak birbirine bağlıyor. Derin uzay keşifleri kapsamında fırlatılan yeni nesil roket ve gelişmiş uydu sistemleri, astronotlar tarafından kontrol edilen teleskop ağlarıyla galaksi haritalarını güncelliyor ve yaşanabilir bir gezegen bulmak adına yörünge operasyonlarını sürdürüyor. Eş zamanlı olarak dünyadaki medya, televizyon, radyo ve gazete gibi geleneksel yayın organlarının yanı sıra sosyal medya platformları üzerinden de bu tarihi anları anlık manşetlerle milyonlara ulaştırıyor. Teknolojik devrimin merkezinde yer alan yapay zeka ve kuantum yazılım algoritmaları, siber güvenlik ağlarındaki şifreleme metotlarını ve veri analitiği süreçlerini kökten değiştirirken, dijital finans piyasalarında blockchain tabanlı kripto para birimlerinin, bitcoin cüzdanlarının ve madencilik faaliyetlerinin hızla yayılmasına öncülük ediyor. Küresel borsa endeksleri, merkez bankalarının enflasyon, faiz ve bütçe kararlarıyla dalgalanırken, sermaye sahipleri yatırım stratejilerini risk yönetimi ve serbest rekabet koşullarına göre yeniden optimize ediyor. Turizm sektörü bu hareketlilikten etkilenerek dijital bilet ve pasaport sistemleriyle seyahat rotalarını esnetiyor; insanlar valizlerini hazırlayıp otel rezervasyonlarını yaparken, meteoroloji uzmanları ise iklim değişikliği kaynaklı ani fırtına, kuraklık ve ekstrem hava durumu uyarılarında bulunuyor. Modern şehirlerde yükselen akıllı gökdelen tasarımları, mimar ve mühendislerin şantiye ortamında geliştirdiği yenilikçi çimento formülleri ve estetik projelerle hayat buluyor. Diğer tarafta arkeologlar, tarih öncesi medeniyetlere ait antik kalıntılar arasında titiz bir kazı yürüterek anıt, fosil ve müzelerde sergilenecek paha biçilemez kronolojik eserler keşfediyor. Üniversite laboratuvarlarında kuantum teorisi, teorik fizik, kimyasal reaksiyonlar ve moleküler biyoloji üzerine derin araştırmalar yapan profesörler, tıp dünyasında çığır açacak yeni bir ilaç ve aşı formülü üzerinde cerrahi hassasiyetle çalışıyor. Hastane koridorlarında doktor ve hemşireler beslenme ve kronik sağlık sorunlarına karşı modern tedavi yöntemleri uygularken, adliye saraylarında hukuk sisteminin temel taşları olan yargıç, savcı ve avukatlar karmaşık bir dava sürecinde anayasa ve kanun maddeleri ışığında adalet arıyor. Akşamları tiyatro, sergi ve senfonik konser etkinliklerinde buluşan sanatseverler, resim, heykel ve müzik eserlerinin felsefi derinliğini tartışırken, gurme şeflerin mutfakta hazırladığı lezzetli menü tarifleri ünlü restoranlarda gastronomi tutkunlarına sunuluyor. Tüm bu karmaşanın uzağında, tarlasında yeni nesil traktör yardımıyla organik tohum ekimi yapan çiftçiler, sulama ve gübreleme teknikleriyle hasat verimliliğini artırmaya çalışırken, doğa dostu mühendisler tarafından kurulan güneş paneli ve rüzgar gülü tesisleri temiz elektrik üreterek karbon emisyonunu ve atık tüketimini azaltmayı, sürdürüebilir bir geleceği inşa etmeyi hedefliyor."""

# ============================================================================
# 1. METNI İŞLE VE KELİMELERİ ÇIKAR
# ============================================================================

def metni_isleme(metin: str) -> List[str]:
    """Metni temizle ve kelimeleri çıkart"""
    # Türkçe karakterleri koru, noktaları ve işaretleri kaldır
    metin = re.sub(r'[,;:\.\!\?\"\(\)«»]', ' ', metin)
    # Türkçe harf korumasıyla küçük harf yap
    metin = metin.lower()
    # Whitespace'e göre böl ve boş olanları filtrele
    kelimeler = [k.strip() for k in metin.split() if k.strip()]
    return kelimeler

# ============================================================================
# 2. KELİMELERİ OTOMATİK KATEGORİZE ET
# ============================================================================

def kelime_kategorize(kelimeler: List[str]) -> Dict[str, List[str]]:
    """Kelimeleri semantic bağlama göre kategorize et"""
    
    # Semantic kategori sözlüğü
    semantic_groups = {
        "uzay_teknoloji": ["uzay", "roket", "uydu", "astronot", "teleskop", "galaksi", "yörünge", "gezegen"],
        "medya": ["medya", "televizyon", "radyo", "gazete", "platform", "manşet", "yayın"],
        "dijital_finans": ["yapay", "zeka", "kuantum", "algoritma", "siber", "şifreleme", "veri", "blockchain", "bitcoin", "madencilik"],
        "ekonomi": ["borsa", "banka", "enflasyon", "faiz", "bütçe", "yatırım", "risk", "rekabet", "sermaye"],
        "turizm": ["turizm", "bilet", "pasaport", "seyahat", "valizler", "otel", "rezervasyon", "meteoroloji"],
        "inşaat_mimarlık": ["şehir", "gökdelen", "mimar", "mühendis", "şantiye", "çimento", "proje", "estetik"],
        "arkeoloji": ["arkeolog", "medeniyetler", "kalıntılar", "kazı", "anıt", "fosil", "müze", "kronolojik"],
        "bilim_araştırma": ["üniversite", "laboratuvar", "kuantum", "teori", "fizik", "kimya", "biyoloji", "deney", "araştırma"],
        "sağlık_tıp": ["hastane", "doktor", "hemşire", "beslenme", "sağlık", "tedavi", "ilaç", "aşı", "cerrahi"],
        "hukuk": ["adliye", "hukuk", "yargıç", "savcı", "avukat", "dava", "anayasa", "kanun", "adalet"],
        "sanat": ["tiyatro", "sergi", "konser", "resim", "heykel", "müzik", "eser", "felsefe"],
        "gastronomi": ["şef", "mutfak", "menü", "restoran", "gastronomi", "lezzetli", "tarif"],
        "tarım_çevre": ["tarım", "traktör", "tohum", "organik", "sulama", "gübreleme", "hasat", "çiftçiler"],
        "enerji": ["güneş", "paneli", "rüzgar", "gülü", "elektrik", "enerji", "karbon", "emisyon", "atık"]
    }
    
    word_pool = {}
    
    for kategori, anahtar_kelimeler in semantic_groups.items():
        word_pool[kategori] = []
        for kelime in kelimeler:
            # Kelimenin herhangi bir anahtar kelimeyi içerip içermediğini kontrol et
            for anahtar in anahtar_kelimeler:
                if anahtar in kelime or kelime in anahtar:
                    if kelime not in word_pool[kategori] and len(kelime) > 2:
                        word_pool[kategori].append(kelime)
                    break
    
    # Kategorileri en az 5 kelime olacak şekilde filtrele
    word_pool = {k: v for k, v in word_pool.items() if len(v) >= 3}
    
    # Boş kategorileri doldur
    tum_kelimeler = set(kelimeler)
    for kategori in list(word_pool.keys()):
        if len(word_pool[kategori]) < 5:
            # Kategoriye ait eksik kelimeleri ekle
            ihtiyac = 5 - len(word_pool[kategori])
            seçenekler = list(tum_kelimeler - set(word_pool[kategori]))
            if seçenekler:
                word_pool[kategori].extend(np.random.choice(seçenekler, size=min(ihtiyac, len(seçenekler)), replace=False).tolist())
    
    return word_pool

# ============================================================================
# 3. CORPUS OLUŞTUR (ORIJINAL KOD YAPISI KULLAN)
# ============================================================================

def generate_sentence(word_pool_dict: Dict, subject_pool: List, verb_pool: List):
    """Orijinal fonksiyon - kelime havuzundan cümle üret"""
    kategori = list(word_pool_dict.keys())
    np.random.shuffle(kategori)
    secilen = []
    
    for k in kategori:
        sayi = np.random.randint(0, 4)
        if sayi > 0 and len(word_pool_dict[k]) > 0:
            secilen.extend(np.random.choice(word_pool_dict[k], size=min(sayi, len(word_pool_dict[k])), replace=False).tolist())
    
    flat_words = []
    for words in word_pool_dict.values():
        flat_words.extend(words)
    flat_words = list(set(flat_words))
    
    if len(secilen) < 3:
        if len(flat_words) > 0:
            secilen.extend(np.random.choice(flat_words, size=min(3, len(flat_words)), replace=False).tolist())
    
    secilen = list(set(secilen))
    
    if len(secilen) < 3 and len(flat_words) > 0:
        secilen = np.random.choice(flat_words, size=min(5, len(flat_words)), replace=False).tolist()
    
    if len(secilen) == 0:
        return "boş cümle"
    
    np.random.shuffle(secilen)
    cumle_uzunlugu = min(np.random.randint(5, 13), len(secilen))
    cumle_kelimeleri = secilen[:cumle_uzunlugu]
    return " ".join(cumle_kelimeleri)

# ============================================================================
# 4. ANA PROGRAM
# ============================================================================

print("=" * 90)
print("METINDEN OTOMATIK WORD_POOL ÇIKARMA VE CORPUS OLUŞTURMA")
print("=" * 90)

# Metni işle
kelimeler = metni_isleme(uzun_metin)
print(f"\n✓ Metindeki kelime sayısı: {len(kelimeler)}")
print(f"✓ Benzersiz kelime sayısı: {len(set(kelimeler))}")

# Kelimeleri kategorize et
word_pool = kelime_kategorize(kelimeler)

print(f"\n✓ Otomatik oluşturulan kategori sayısı: {len(word_pool)}")
print("\n" + "=" * 90)
print("KATEGORİLER VE KELİMELER:")
print("=" * 90)

for kategori, kelime_listesi in sorted(word_pool.items()):
    print(f"\n  {kategori:25s} ({len(kelime_listesi)} kelime):")
    print(f"    {', '.join(kelime_listesi[:10])}")
    if len(kelime_listesi) > 10:
        print(f"    ... ve {len(kelime_listesi) - 10} daha")

# Konu havuzu oluştur (kategorilerdeki ilk kelimeler + bazı ekler)
subject_pool = []
for kelime_listesi in word_pool.values():
    subject_pool.extend(kelime_listesi[:2])  # Her kategoriden ilk 2 kelimeyi al
subject_pool = list(set(subject_pool))[:20]  # Çeşitliliği sınırla

# Eylem havuzu (orijinal)
verb_pool = ["yapar", "eder", "olur", "gelir", "gider", "calisir", "yazar", "okur", 
             "söyler", "dinler", "izler", "basar", "kazanir", "kaybeder", "verir", "alir", 
             "korur", "savunur", "yürütür", "yönetir"]

print(f"\n✓ Oluşturulan konu havuzu boyutu: {len(subject_pool)}")
print(f"✓ Eylem havuzu boyutu: {len(verb_pool)}")

# Corpus oluştur
print(f"\n{'=' * 90}")
print("CORPUS OLUŞTURULUYOR...")
print(f"{'=' * 90}\n")

korpus = [generate_sentence(word_pool, subject_pool, verb_pool) for _ in range(2000)]

# İstatistikler
tum_kelimeler = ' '.join(korpus).split()
benzersiz_kelimeler = set(tum_kelimeler)

print(f"✓ Corpus boyutu: {len(korpus)} cümle")
print(f"✓ Benzersiz kelime sayısı: {len(benzersiz_kelimeler)}")
print(f"✓ Toplam kelime sayısı: {len(tum_kelimeler)}")
print(f"✓ Ortalama cümle uzunluğu: {np.mean([len(c.split()) for c in korpus]):.1f} kelime")

print("\n" + "=" * 90)
print("İLK 10 CÜMLE:")
print("=" * 90)
for i, doc in enumerate(korpus[:10], 1):
    print(f"{i:2d}. {doc}")

print("\n" + "=" * 90)
print("EN SIK 15 KELIME:")
print("=" * 90)
kelime_sayilari = Counter(tum_kelimeler)
for kelime, sayi in kelime_sayilari.most_common(15):
    print(f"  {kelime:30s} : {sayi:4d} kez")

# Dosyaya kaydet
with open('/home/claude/corpus_otomatik.txt', 'w', encoding='utf-8') as f:
    for i, cumle in enumerate(korpus, 1):
        f.write(f"{i}\t{cumle}\n")

print(f"\n✓ Corpus 'corpus_otomatik.txt' dosyasına kaydedildi")

# Word pool'u dosyaya kaydet
with open('/home/claude/word_pool_otomatik.txt', 'w', encoding='utf-8') as f:
    f.write("# Otomatik Çıkartılan Word Pool\n\n")
    for kategori, kelimeler in sorted(word_pool.items()):
        f.write(f'"{kategori}": {kelimeler},\n')

print(f"✓ Word Pool 'word_pool_otomatik.txt' dosyasına kaydedildi")
print("\n" + "=" * 90)