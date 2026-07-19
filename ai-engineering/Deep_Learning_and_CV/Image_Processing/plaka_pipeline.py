# -*- coding: utf-8 -*-
"""
GÖRÜNTÜ İŞLEME — İLERİ SEVİYE
Plaka Tespiti Verisinden İkili Sınıflandırma Pipeline'ı ("car-license-plate" vs "plaka-değil")

Veri seti: COCO formatlı plaka tespit verisi (data/train, data/valid, data/test
altında görseller + _annotations.coco.json). Veride sadece TEK kategori var
("car-license-plate"), yani il/tip gibi bir sınıflandırma etiketi yok. Bu yüzden
bounding box'lardan PLAKA (pozitif) ve PLAKA-OLMAYAN (negatif) bölgeler kesilerek
ikili sınıflandırma veri seti oluşturuluyor; 9 bölümün tamamı bu veri üzerinde
uygulanıyor.

Bölümler:
 13 - Veri Hazırlama (COCO -> crop) + Veri Artırma
 14 - PCA ile Boyut İndirgeme + Görselleştirme
 15 - t-SNE ile Özellik Uzayı Keşfi
 16 - Konvolüsyon Katmanı — Sıfırdan NumPy
 17 - Aktivasyon Fonksiyonları ve Özellik Haritaları
 18 - Tam Bağlı Sinir Ağı (MLP) ile Sınıflandırma
 19 - SVM vs MLP vs KNN — Kapsamlı Karşılaştırma
 20 - Grad-CAM Benzeri Isı Haritası (Occlusion Sensitivity)
 21 - Gerçek CNN Pipeline — Sıfırdan İleri Yayılım

Çalıştırma:
    python3 plaka_pipeline.py
Proje kök dizininizde çalıştırın (data/ ve outputs/ klasörleriyle aynı seviyede).
"""

import os
import json
import random

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # ekran olmadan da PNG üretebilmek için
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)

# --------------------------------------------------------------------------
# GENEL AYARLAR
# --------------------------------------------------------------------------
DATA_DIR = "data"
OUT_DIR = "outputs"
AUG_DIR = os.path.join(OUT_DIR, "augmented_data")
IMG_SIZE = 32          # her crop bu boyuta (IMG_SIZE x IMG_SIZE, gri tonlama) resize edilir
NEG_PER_IMAGE = 2       # görsel başına üretilecek negatif (plaka-olmayan) crop sayısı
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(AUG_DIR, exist_ok=True)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ==========================================================================
# BÖLÜM 13 — VERİ HAZIRLAMA (COCO -> crop) + VERİ ARTIRMA
# ==========================================================================

def iou(box_a, box_b):
    """İki (x1,y1,x2,y2) kutusu arasındaki IoU (kesişim/birleşim) oranı."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_coco(json_path):
    """COCO json'unu image_id -> dosya adı ve image_id -> [bbox,...] eşlemelerine çevirir."""
    d = json.load(open(json_path, "r", encoding="utf-8"))
    img_map = {im["id"]: im["file_name"] for im in d["images"]}
    boxes_map = {}
    for ann in d["annotations"]:
        boxes_map.setdefault(ann["image_id"], []).append(ann["bbox"])
    return img_map, boxes_map


def build_dataset(split_dir, img_size=IMG_SIZE, neg_per_image=NEG_PER_IMAGE, seed=RANDOM_STATE):
    """
    Bir split klasöründen (data/train, data/valid, data/test) ikili sınıflandırma
    veri seti üretir.
    Döndürür: X (N, img_size, img_size) float32 [0,1], y (N,) {0,1}
              1 = plaka, 0 = plaka değil
    """
    json_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"COCO annotasyon dosyası bulunamadı: {json_path}")

    rng = random.Random(seed)
    img_map, boxes_map = load_coco(json_path)
    X, y = [], []

    for image_id, fname in img_map.items():
        path = os.path.join(split_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pos_boxes_xyxy = []
        for bbox in boxes_map.get(image_id, []):
            x, ytop, w_, h = bbox
            x1, y1 = max(0, int(x)), max(0, int(ytop))
            x2, y2 = min(W, int(x + w_)), min(H, int(ytop + h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(gray[y1:y2, x1:x2], (img_size, img_size))
            X.append(crop.astype(np.float32) / 255.0)
            y.append(1)
            pos_boxes_xyxy.append((x1, y1, x2, y2))

        # negatif (plaka-olmayan) bölgeler: pozitiflerle IoU < 0.1 olacak şekilde rastgele kutu
        n_neg, attempts = 0, 0
        while n_neg < neg_per_image and attempts < 25:
            attempts += 1
            bw = rng.randint(20, max(21, W // 4))
            bh = rng.randint(10, max(11, H // 4))
            if W - bw <= 0 or H - bh <= 0:
                break
            nx1, ny1 = rng.randint(0, W - bw), rng.randint(0, H - bh)
            nx2, ny2 = nx1 + bw, ny1 + bh
            if all(iou((nx1, ny1, nx2, ny2), pb) < 0.1 for pb in pos_boxes_xyxy):
                crop = cv2.resize(gray[ny1:ny2, nx1:nx2], (img_size, img_size))
                X.append(crop.astype(np.float32) / 255.0)
                y.append(0)
                n_neg += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# --- Veri Artırma fonksiyonları (numpy tabanlı, [0,1] aralığında gri görüntü) ---

def augment_flip(img):
    return np.fliplr(img)

def augment_rotate(img, angle_deg):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def augment_brightness(img, factor):
    return np.clip(img * factor, 0, 1)

def augment_noise(img, std=0.05):
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 1)

def augment_shift(img, dx, dy):
    h, w = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_random_augmentation(img):
    """Bir görüntüye rastgele 1-3 artırma tekniği uygular."""
    ops = [
        lambda im: augment_flip(im),
        lambda im: augment_rotate(im, random.uniform(-15, 15)),
        lambda im: augment_brightness(im, random.uniform(0.7, 1.3)),
        lambda im: augment_noise(im, 0.04),
        lambda im: augment_shift(im, random.randint(-3, 3), random.randint(-3, 3)),
    ]
    chosen = random.sample(ops, k=random.randint(1, 2))
    out = img.copy()
    for op in chosen:
        out = op(out)
    return out


def section13_prepare_and_augment():
    print("\n=== BÖLÜM 13: Veri Hazırlama + Veri Artırma ===")

    X_train, y_train = build_dataset(os.path.join(DATA_DIR, "train"))
    X_valid, y_valid = build_dataset(os.path.join(DATA_DIR, "valid"))
    X_test, y_test = build_dataset(os.path.join(DATA_DIR, "test"))

    print(f"Train: {X_train.shape}, pozitif={int(y_train.sum())}, negatif={int((y_train==0).sum())}")
    print(f"Valid: {X_valid.shape}, pozitif={int(y_valid.sum())}, negatif={int((y_valid==0).sum())}")
    print(f"Test : {X_test.shape}, pozitif={int(y_test.sum())}, negatif={int((y_test==0).sum())}")

    if len(X_train) == 0:
        raise RuntimeError(
            "Train setinden hiç görüntü/crop üretilemedi. DATA_DIR yolunu ve "
            "_annotations.coco.json dosyasının varlığını kontrol edin."
        )

    # Birkaç örnek artırma görselleştirmesi kaydet
    n_show = min(4, len(X_train))
    fig, axes = plt.subplots(n_show, 4, figsize=(10, 2.5 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    for i in range(n_show):
        orig = X_train[i]
        axes[i, 0].imshow(orig, cmap="gray"); axes[i, 0].set_title("Orijinal"); axes[i, 0].axis("off")
        axes[i, 1].imshow(augment_flip(orig), cmap="gray"); axes[i, 1].set_title("Flip"); axes[i, 1].axis("off")
        axes[i, 2].imshow(augment_rotate(orig, 12), cmap="gray"); axes[i, 2].set_title("Rotate"); axes[i, 2].axis("off")
        axes[i, 3].imshow(augment_noise(orig), cmap="gray"); axes[i, 3].set_title("Noise"); axes[i, 3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(AUG_DIR, "13_veri_artirma_ornekleri.png"), dpi=120)
    plt.close(fig)

    # Eğitim setini artırılmış kopyalarla genişlet (her örnek için 1 artırılmış kopya ekle)
    aug_X = [apply_random_augmentation(img) for img in X_train]
    X_train_aug = np.concatenate([X_train, np.array(aug_X)], axis=0)
    y_train_aug = np.concatenate([y_train, y_train], axis=0)
    print(f"Artırma sonrası eğitim seti: {X_train_aug.shape}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_train_aug": X_train_aug, "y_train_aug": y_train_aug,
        "X_valid": X_valid, "y_valid": y_valid,
        "X_test": X_test, "y_test": y_test,
    }


# ==========================================================================
# BÖLÜM 14 — PCA İLE BOYUT İNDİRGEME + GÖRSELLEŞTİRME
# ==========================================================================

def section14_pca(data):
    print("\n=== BÖLÜM 14: PCA ile Boyut İndirgeme ===")
    X_flat = data["X_train_aug"].reshape(len(data["X_train_aug"]), -1)
    y = data["y_train_aug"]

    n_components = min(50, X_flat.shape[0], X_flat.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_flat)

    # Açıklanan varyans grafiği
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", markersize=3)
    plt.xlabel("Bileşen Sayısı")
    plt.ylabel("Kümülatif Açıklanan Varyans")
    plt.title("PCA — Kümülatif Açıklanan Varyans (Plaka Verisi)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "14_pca_varyans.png"), dpi=120)
    plt.close()

    # "Eigenplates" — ilk birkaç temel bileşenin görüntü olarak gösterimi
    n_eig = min(8, pca.components_.shape[0])
    fig, axes = plt.subplots(2, n_eig // 2, figsize=(2 * (n_eig // 2), 4))
    for i, ax in enumerate(axes.ravel()):
        comp = pca.components_[i].reshape(IMG_SIZE, IMG_SIZE)
        ax.imshow(comp, cmap="gray")
        ax.set_title(f"PC {i+1}")
        ax.axis("off")
    plt.suptitle("Eigenplates — İlk Temel Bileşenler")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "14_eigenplates.png"), dpi=120)
    plt.close()

    # İlk 2 bileşenle sınıf dağılımı
    plt.figure(figsize=(6, 5))
    for label, color, name in [(0, "tab:red", "plaka-değil"), (1, "tab:blue", "plaka")]:
        mask = y == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.6, c=color, label=name)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA — İlk 2 Bileşen Üzerinde Sınıf Dağılımı")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "14_pca_2d_dagilim.png"), dpi=120)
    plt.close()

    print(f"PCA tamamlandı. İlk 2 bileşen açıklanan varyans: {pca.explained_variance_ratio_[:2].sum():.3f}")
    return pca, X_pca


# ==========================================================================
# BÖLÜM 15 — t-SNE İLE ÖZELLİK UZAYI KEŞFİ
# ==========================================================================

def section15_tsne(data, pca):
    print("\n=== BÖLÜM 15: t-SNE ile Özellik Uzayı Keşfi ===")
    X_flat = data["X_train_aug"].reshape(len(data["X_train_aug"]), -1)
    y = data["y_train_aug"]

    # Önce PCA ile ön-indirgeme (t-SNE için standart pratik), sonra t-SNE
    X_reduced = pca.transform(X_flat)
    n_components_pre = min(30, X_reduced.shape[1])
    X_reduced = X_reduced[:, :n_components_pre]

    perplexity = min(30, max(5, len(X_reduced) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE, init="pca")
    X_tsne = tsne.fit_transform(X_reduced)

    plt.figure(figsize=(6, 5))
    for label, color, name in [(0, "tab:red", "plaka-değil"), (1, "tab:blue", "plaka")]:
        mask = y == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=12, alpha=0.6, c=color, label=name)
    plt.title("t-SNE — Plaka / Plaka-Değil Özellik Uzayı Haritası")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "15_tsne_haritasi.png"), dpi=120)
    plt.close()

    print("t-SNE haritası kaydedildi.")
    return X_tsne


# ==========================================================================
# BÖLÜM 16 — KONVOLÜSYON KATMANI — SIFIRDAN NUMPY
# ==========================================================================

def conv2d(img, kernel, stride=1):
    """Tek kanallı bir görüntüye 2D konvolüsyon uygular (sıfırdan, NumPy)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out_h = (img.shape[0] - 1) // stride + 1
    out_w = (img.shape[1] - 1) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for oi, i in enumerate(range(0, img.shape[0], stride)):
        for oj, j in enumerate(range(0, img.shape[1], stride)):
            region = padded[i:i + kh, j:j + kw]
            out[oi, oj] = np.sum(region * kernel)
    return out


KERNELS = {
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "blur": np.ones((3, 3), dtype=np.float32) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
}


def section16_convolution(data):
    print("\n=== BÖLÜM 16: Konvolüsyon Katmanı — Sıfırdan NumPy ===")
    sample = data["X_train"][int(np.argmax(data["y_train"]))]  # bir pozitif (plaka) örneği seç

    fig, axes = plt.subplots(1, len(KERNELS) + 1, figsize=(3 * (len(KERNELS) + 1), 3))
    axes[0].imshow(sample, cmap="gray"); axes[0].set_title("Orijinal"); axes[0].axis("off")
    feature_maps = {}
    for ax, (name, kernel) in zip(axes[1:], KERNELS.items()):
        fmap = conv2d(sample, kernel)
        feature_maps[name] = fmap
        ax.imshow(fmap, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    plt.suptitle("Sıfırdan Konvolüsyon — Manuel Çekirdekler")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "16_konvolusyon_ornekleri.png"), dpi=120)
    plt.close()

    print("Konvolüsyon çıktıları kaydedildi.")
    return sample, feature_maps


# ==========================================================================
# BÖLÜM 17 — AKTİVASYON FONKSİYONLARI VE ÖZELLİK HARİTALARI
# ==========================================================================

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh_act(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


ACTIVATIONS = {"ReLU": relu, "Sigmoid": sigmoid, "Tanh": tanh_act, "LeakyReLU": leaky_relu}


def section17_activations(sample, feature_maps):
    print("\n=== BÖLÜM 17: Aktivasyon Fonksiyonları ===")

    # Aktivasyon eğrileri
    x = np.linspace(-5, 5, 200)
    plt.figure(figsize=(6, 4))
    for name, fn in ACTIVATIONS.items():
        plt.plot(x, fn(x), label=name)
    plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
    plt.legend(); plt.title("Aktivasyon Fonksiyonları")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "17_aktivasyon_egrileri.png"), dpi=120)
    plt.close()

    # sobel_x özellik haritasına aktivasyon uygulanmış hali
    fmap = feature_maps["sobel_x"]
    fig, axes = plt.subplots(1, len(ACTIVATIONS) + 1, figsize=(3 * (len(ACTIVATIONS) + 1), 3))
    axes[0].imshow(fmap, cmap="gray"); axes[0].set_title("Ham Özellik Haritası (sobel_x)"); axes[0].axis("off")
    for ax, (name, fn) in zip(axes[1:], ACTIVATIONS.items()):
        ax.imshow(fn(fmap), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    plt.suptitle("Aktivasyon Sonrası Özellik Haritaları")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "17_aktivasyon_sonrasi_haritalar.png"), dpi=120)
    plt.close()

    print("Aktivasyon görselleri kaydedildi.")


# ==========================================================================
# BÖLÜM 18 — TAM BAĞLI SİNİR AĞI (MLP) İLE SINIFLANDIRMA
# ==========================================================================

def section18_mlp(data):
    print("\n=== BÖLÜM 18: MLP ile Sınıflandırma ===")
    X_train = data["X_train_aug"].reshape(len(data["X_train_aug"]), -1)
    y_train = data["y_train_aug"]
    X_valid = data["X_valid"].reshape(len(data["X_valid"]), -1)
    y_valid = data["y_valid"]
    X_test = data["X_test"].reshape(len(data["X_test"]), -1)
    y_test = data["y_test"]

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train, y_train)

    for split_name, X_s, y_s in [("Valid", X_valid, y_valid), ("Test", X_test, y_test)]:
        if len(X_s) == 0:
            continue
        y_pred = mlp.predict(X_s)
        acc = accuracy_score(y_s, y_pred)
        print(f"MLP {split_name} doğruluk: {acc:.3f}")

    if len(X_test) > 0:
        y_pred_test = mlp.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(cm, display_labels=["plaka-değil", "plaka"])
        disp.plot(cmap="Blues")
        plt.title("MLP — Test Seti Karışıklık Matrisi")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "18_mlp_confusion_matrix.png"), dpi=120)
        plt.close()

    return mlp


# ==========================================================================
# BÖLÜM 19 — SVM vs MLP vs KNN — KAPSAMLI KARŞILAŞTIRMA
# ==========================================================================

def section19_compare_models(data, mlp):
    print("\n=== BÖLÜM 19: SVM vs MLP vs KNN Karşılaştırma ===")
    X_train = data["X_train_aug"].reshape(len(data["X_train_aug"]), -1)
    y_train = data["y_train_aug"]
    X_test = data["X_test"].reshape(len(data["X_test"]), -1)
    y_test = data["y_test"]

    if len(X_test) == 0:
        print("Test seti boş, karşılaştırma atlanıyor.")
        return {}

    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    results = {}
    for name, model in [("SVM", svm), ("MLP", mlp), ("KNN", knn)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"{name}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")

    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(results):
        vals = [results[name][m] for m in metrics]
        plt.bar(x + i * width, vals, width, label=name)
    plt.xticks(x + width, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Skor")
    plt.title("SVM vs MLP vs KNN — Test Seti Karşılaştırması")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "19_model_karsilastirma.png"), dpi=120)
    plt.close()

    return {"svm": svm, "knn": knn, "results": results}


# ==========================================================================
# BÖLÜM 20 — GRAD-CAM BENZERİ ISI HARİTASI (OCCLUSION SENSITIVITY)
# ==========================================================================

def occlusion_saliency(model, img, patch_size=6, stride=3):
    """
    Gerçek Grad-CAM, eğitilmiş bir CNN'in gradyanlarına ihtiyaç duyar; burada
    framework bağımsız (numpy + sklearn modeliyle çalışan) bir alternatif olarak
    "occlusion sensitivity" kullanılıyor: görüntü üzerinde küçük bir yama
    kaydırılıp örtülüyor, modelin "plaka" olasılığındaki düşüş ısı haritasına
    yazılıyor. Yüksek düşüş = model o bölgeye çok güveniyor demek.
    """
    h, w = img.shape
    base_prob = model.predict_proba(img.reshape(1, -1))[0, 1]
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            occluded = img.copy()
            occluded[i:i + patch_size, j:j + patch_size] = 0.5
            prob = model.predict_proba(occluded.reshape(1, -1))[0, 1]
            drop = max(0.0, base_prob - prob)
            heatmap[i:i + patch_size, j:j + patch_size] += drop
            counts[i:i + patch_size, j:j + patch_size] += 1

    counts[counts == 0] = 1
    heatmap = heatmap / counts
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap, base_prob


def section20_saliency(data, mlp):
    print("\n=== BÖLÜM 20: Grad-CAM Benzeri Isı Haritası ===")
    pos_idx = np.where(data["y_test"] == 1)[0]
    if len(pos_idx) == 0:
        pos_idx = np.where(data["y_train"] == 1)[0]
        sample_img = data["X_train"][pos_idx[0]] if len(pos_idx) else data["X_train"][0]
    else:
        sample_img = data["X_test"][pos_idx[0]]

    heatmap, base_prob = occlusion_saliency(mlp, sample_img)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(sample_img, cmap="gray"); axes[0].set_title("Orijinal"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Isı Haritası"); axes[1].axis("off")
    axes[2].imshow(sample_img, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay (p={base_prob:.2f})")
    axes[2].axis("off")
    plt.suptitle("Occlusion Sensitivity — Grad-CAM Benzeri Görselleştirme")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "20_saliency_heatmap.png"), dpi=120)
    plt.close()

    print(f"Isı haritası kaydedildi. Modelin bu örnek için 'plaka' olasılığı: {base_prob:.3f}")


# ==========================================================================
# BÖLÜM 21 — GERÇEK CNN PIPELINE — SIFIRDAN İLERİ YAYILIM
# ==========================================================================

def max_pool2d(fmap, pool_size=2, stride=2):
    h, w = fmap.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            region = fmap[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i, j] = np.max(region)
    return out


def cnn_forward_pass(img, kernels):
    """
    Basit bir CNN'in ileri yayılımını sıfırdan gösterir:
    Conv -> ReLU -> MaxPool -> Flatten -> Dense (rastgele ağırlıklarla, gösterim amaçlı)
    """
    conv_outputs = [conv2d(img, k) for k in kernels.values()]
    relu_outputs = [relu(c) for c in conv_outputs]
    pooled_outputs = [max_pool2d(r) for r in relu_outputs]

    flattened = np.concatenate([p.flatten() for p in pooled_outputs])
    rng = np.random.RandomState(RANDOM_STATE)
    W_dense = rng.randn(flattened.shape[0], 2) * 0.01
    b_dense = np.zeros(2)
    logits = flattened @ W_dense + b_dense
    probs = np.exp(logits) / np.sum(np.exp(logits))

    return {
        "conv": conv_outputs,
        "relu": relu_outputs,
        "pooled": pooled_outputs,
        "flattened": flattened,
        "probs": probs,
    }


def section21_cnn_pipeline(data):
    print("\n=== BÖLÜM 21: Gerçek CNN Pipeline — Sıfırdan İleri Yayılım ===")
    pos_idx = np.where(data["y_train"] == 1)[0]
    sample = data["X_train"][pos_idx[0]] if len(pos_idx) else data["X_train"][0]

    result = cnn_forward_pass(sample, KERNELS)

    n_k = len(KERNELS)
    fig, axes = plt.subplots(3, n_k, figsize=(3 * n_k, 8))
    kernel_names = list(KERNELS.keys())
    for j, name in enumerate(kernel_names):
        axes[0, j].imshow(result["conv"][j], cmap="gray"); axes[0, j].set_title(f"Conv: {name}"); axes[0, j].axis("off")
        axes[1, j].imshow(result["relu"][j], cmap="gray"); axes[1, j].set_title("ReLU"); axes[1, j].axis("off")
        axes[2, j].imshow(result["pooled"][j], cmap="gray"); axes[2, j].set_title("MaxPool"); axes[2, j].axis("off")
    plt.suptitle("Sıfırdan CNN İleri Yayılım: Conv -> ReLU -> MaxPool")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "21_cnn_pipeline_ileri_yayilim.png"), dpi=120)
    plt.close()

    print(f"Flatten boyutu: {result['flattened'].shape[0]}")
    print(f"(Eğitilmemiş, rastgele ağırlıklı) çıkış olasılıkları: {result['probs']}")


# ==========================================================================
# ANA AKIŞ
# ==========================================================================

# -*- coding: utf-8 -*-
"""
GÖRÜNTÜ İŞLEME — İLERİ SEVİYE
Plaka Tespiti Verisinden İkili Sınıflandırma Pipeline'ı ("car-license-plate" vs "plaka-değil")

Veri seti: COCO formatlı plaka tespit verisi (data/train, data/valid, data/test
altında görseller + _annotations.coco.json). Veride sadece TEK kategori var
("car-license-plate"), yani il/tip gibi bir sınıflandırma etiketi yok. Bu yüzden
bounding box'lardan PLAKA (pozitif) ve PLAKA-OLMAYAN (negatif) bölgeler kesilerek
ikili sınıflandırma veri seti oluşturuluyor; 9 bölümün tamamı bu veri üzerinde
uygulanıyor.

Bölümler:
 13 - Veri Hazırlama (COCO -> crop) + Veri Artırma
 14 - PCA ile Boyut İndirgeme + Görselleştirme
 15 - t-SNE ile Özellik Uzayı Keşfi
 16 - Konvolüsyon Katmanı — Sıfırdan NumPy
 17 - Aktivasyon Fonksiyonları ve Özellik Haritaları
 18 - Tam Bağlı Sinir Ağı (MLP) ile Sınıflandırma
 19 - SVM vs MLP vs KNN — Kapsamlı Karşılaştırma
 20 - Grad-CAM Benzeri Isı Haritası (Occlusion Sensitivity)
 21 - Gerçek CNN Pipeline — Sıfırdan İleri Yayılım

Çalıştırma:
    python3 plaka_pipeline.py
Proje kök dizininizde çalıştırın (data/ ve outputs/ klasörleriyle aynı seviyede).
"""

import os
import json
import random

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # ekran olmadan da PNG üretebilmek için
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)

# --------------------------------------------------------------------------
# GENEL AYARLAR
# --------------------------------------------------------------------------
DATA_DIR = "data"
OUT_DIR = "outputs"
AUG_DIR = os.path.join(OUT_DIR, "augmented_data")
IMG_SIZE = 32          # her crop bu boyuta (IMG_SIZE x IMG_SIZE, gri tonlama) resize edilir
NEG_PER_IMAGE = 2       # görsel başına üretilecek negatif (plaka-olmayan) crop sayısı
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(AUG_DIR, exist_ok=True)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ==========================================================================
# BÖLÜM 13 — VERİ HAZIRLAMA (COCO -> crop) + VERİ ARTIRMA
# ==========================================================================

def iou(box_a, box_b):
    """İki (x1,y1,x2,y2) kutusu arasındaki IoU (kesişim/birleşim) oranı."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_coco(json_path):
    """COCO json'unu image_id -> dosya adı ve image_id -> [bbox,...] eşlemelerine çevirir."""
    d = json.load(open(json_path, "r", encoding="utf-8"))
    img_map = {im["id"]: im["file_name"] for im in d["images"]}
    boxes_map = {}
    for ann in d["annotations"]:
        boxes_map.setdefault(ann["image_id"], []).append(ann["bbox"])
    return img_map, boxes_map


def build_dataset(split_dir, img_size=IMG_SIZE, neg_per_image=NEG_PER_IMAGE, seed=RANDOM_STATE):
    """
    Bir split klasöründen (data/train, data/valid, data/test) ikili sınıflandırma
    veri seti üretir.
    Döndürür: X (N, img_size, img_size) float32 [0,1], y (N,) {0,1}
              1 = plaka, 0 = plaka değil
    """
    json_path = os.path.join(split_dir, "_annotations.coco.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"COCO annotasyon dosyası bulunamadı: {json_path}")

    rng = random.Random(seed)
    img_map, boxes_map = load_coco(json_path)
    X, y = [], []

    for image_id, fname in img_map.items():
        path = os.path.join(split_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pos_boxes_xyxy = []
        for bbox in boxes_map.get(image_id, []):
            x, ytop, w_, h = bbox
            x1, y1 = max(0, int(x)), max(0, int(ytop))
            x2, y2 = min(W, int(x + w_)), min(H, int(ytop + h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(gray[y1:y2, x1:x2], (img_size, img_size))
            X.append(crop.astype(np.float32) / 255.0)
            y.append(1)
            pos_boxes_xyxy.append((x1, y1, x2, y2))

        # negatif (plaka-olmayan) bölgeler: pozitiflerle IoU < 0.1 olacak şekilde rastgele kutu
        n_neg, attempts = 0, 0
        while n_neg < neg_per_image and attempts < 25:
            attempts += 1
            bw = rng.randint(20, max(21, W // 4))
            bh = rng.randint(10, max(11, H // 4))
            if W - bw <= 0 or H - bh <= 0:
                break
            nx1, ny1 = rng.randint(0, W - bw), rng.randint(0, H - bh)
            nx2, ny2 = nx1 + bw, ny1 + bh
            if all(iou((nx1, ny1, nx2, ny2), pb) < 0.1 for pb in pos_boxes_xyxy):
                crop = cv2.resize(gray[ny1:ny2, nx1:nx2], (img_size, img_size))
                X.append(crop.astype(np.float32) / 255.0)
                y.append(0)
                n_neg += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# --- Veri Artırma fonksiyonları (numpy tabanlı, [0,1] aralığında gri görüntü) ---

def augment_flip(img):
    return np.fliplr(img)

def augment_rotate(img, angle_deg):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def augment_brightness(img, factor):
    return np.clip(img * factor, 0, 1)

def augment_noise(img, std=0.05):
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 1)

def augment_shift(img, dx, dy):
    h, w = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_random_augmentation(img):
    """Bir görüntüye rastgele 1-3 artırma tekniği uygular."""
    ops = [
        lambda im: augment_flip(im),
        lambda im: augment_rotate(im, random.uniform(-15, 15)),
        lambda im: augment_brightness(im, random.uniform(0.7, 1.3)),
        lambda im: augment_noise(im, 0.04),
        lambda im: augment_shift(im, random.randint(-3, 3), random.randint(-3, 3)),
    ]
    chosen = random.sample(ops, k=random.randint(1, 2))
    out = img.copy()
    for op in chosen:
        out = op(out)
    return out


def section13_prepare_and_augment():
    print("\n=== BÖLÜM 13: Veri Hazırlama + Veri Artırma ===")

    X_train, y_train = build_dataset(os.path.join(DATA_DIR, "train"))
    X_valid, y_valid = build_dataset(os.path.join(DATA_DIR, "valid"))
    X_test, y_test = build_dataset(os.path.join(DATA_DIR, "test"))

    print(f"Train: {X_train.shape}, pozitif={int(y_train.sum())}, negatif={int((y_train==0).sum())}")
    print(f"Valid: {X_valid.shape}, pozitif={int(y_valid.sum())}, negatif={int((y_valid==0).sum())}")
    print(f"Test : {X_test.shape}, pozitif={int(y_test.sum())}, negatif={int((y_test==0).sum())}")

    if len(X_train) == 0:
        raise RuntimeError(
            "Train setinden hiç görüntü/crop üretilemedi. DATA_DIR yolunu ve "
            "_annotations.coco.json dosyasının varlığını kontrol edin."
        )

    # Birkaç örnek artırma görselleştirmesi kaydet
    n_show = min(4, len(X_train))
    fig, axes = plt.subplots(n_show, 4, figsize=(10, 2.5 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    for i in range(n_show):
        orig = X_train[i]
        axes[i, 0].imshow(orig, cmap="gray"); axes[i, 0].set_title("Orijinal"); axes[i, 0].axis("off")
        axes[i, 1].imshow(augment_flip(orig), cmap="gray"); axes[i, 1].set_title("Flip"); axes[i, 1].axis("off")
        axes[i, 2].imshow(augment_rotate(orig, 12), cmap="gray"); axes[i, 2].set_title("Rotate"); axes[i, 2].axis("off")
        axes[i, 3].imshow(augment_noise(orig), cmap="gray"); axes[i, 3].set_title("Noise"); axes[i, 3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(AUG_DIR, "13_veri_artirma_ornekleri.png"), dpi=120)
    plt.close(fig)

    # Eğitim setini artırılmış kopyalarla genişlet (her örnek için 1 artırılmış kopya ekle)
    aug_X = [apply_random_augmentation(img) for img in X_train]
    X_train_aug = np.concatenate([X_train, np.array(aug_X)], axis=0)
    y_train_aug = np.concatenate([y_train, y_train], axis=0)
    print(f"Artırma sonrası eğitim seti: {X_train_aug.shape}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_train_aug": X_train_aug, "y_train_aug": y_train_aug,
        "X_valid": X_valid, "y_valid": y_valid,
        "X_test": X_test, "y_test": y_test,
    }


# ==========================================================================
# BÖLÜM 14 — PCA İLE BOYUT İNDİRGEME + GÖRSELLEŞTİRME
# ==========================================================================

def section14_pca(data):
    print("\n=== BÖLÜM 14: PCA ile Boyut İndirgeme ===")
    X_flat = data["X_train_aug"].reshape(len(data["X_train_aug"]), -1)
    y = data["y_train_aug"]

    n_components = min(50, X_flat.shape[0], X_flat.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_flat)

    # Açıklanan varyans grafiği
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", markersize=3)
    plt.xlabel("Bileşen Sayısı")
    plt.ylabel("Kümülatif Açıklanan Varyans")
    plt.title("PCA — Kümülatif Açıklanan Varyans (Plaka Verisi)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "14_pca_varyans.png"), dpi=120)
    plt.close()

    # "Eigenplates" — ilk birkaç temel bileşenin görüntü olarak gösterimi
    n_eig = min(8, pca.components_.shape[0])
    fig, axes = plt.subplots(2, n_eig // 2, figsize=(2 * (n_eig // 2), 4))
    for i, ax in enumerate(axes.ravel()):
        comp = pca.components_[i].reshape(IMG_SIZE, IMG_SIZE)
        ax.imshow(comp, cmap="gray")
        ax.set_title(f"PC {i+1}")
        ax.axis("off")
    plt.suptitle("Eigenplates — İlk Temel Bileşenler")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "14_eigenplates.png"), dpi=120)
    plt.close()

    # İlk 2 bileşenle sınıf dağılımı
    plt.figure(figsize=(6, 5))
    for label, color, name in [(0, "tab:red", "plaka-değil"), (1, "tab:blue", "plaka")]:
        mask = y == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.6, c=color, label=name)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA — İlk 2 Bileşen Üzerinde Sınıf Dağılımı")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "14_pca_2d_dagilim.png"), dpi=120)
    plt.close()

    print(f"PCA tamamlandı. İlk 2 bileşen açıklanan varyans: {pca.explained_variance_ratio_[:2].sum():.3f}")
    return pca, X_pca


# ==========================================================================
# BÖLÜM 15 — t-SNE İLE ÖZELLİK UZAYI KEŞFİ
# ==========================================================================

def section15_tsne(data, pca):
    print("\n=== BÖLÜM 15: t-SNE ile Özellik Uzayı Keşfi ===")
    X_flat = data["X_train_aug"].reshape(len(data["X_train_aug"]), -1)
    y = data["y_train_aug"]

    # Önce PCA ile ön-indirgeme (t-SNE için standart pratik), sonra t-SNE
    X_reduced = pca.transform(X_flat)
    n_components_pre = min(30, X_reduced.shape[1])
    X_reduced = X_reduced[:, :n_components_pre]

    perplexity = min(30, max(5, len(X_reduced) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE, init="pca")
    X_tsne = tsne.fit_transform(X_reduced)

    plt.figure(figsize=(6, 5))
    for label, color, name in [(0, "tab:red", "plaka-değil"), (1, "tab:blue", "plaka")]:
        mask = y == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=12, alpha=0.6, c=color, label=name)
    plt.title("t-SNE — Plaka / Plaka-Değil Özellik Uzayı Haritası")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "15_tsne_haritasi.png"), dpi=120)
    plt.close()

    print("t-SNE haritası kaydedildi.")
    return X_tsne


# ==========================================================================
# BÖLÜM 16 — KONVOLÜSYON KATMANI — SIFIRDAN NUMPY
# ==========================================================================

def conv2d(img, kernel, stride=1):
    """Tek kanallı bir görüntüye 2D konvolüsyon uygular (sıfırdan, NumPy)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out_h = (img.shape[0] - 1) // stride + 1
    out_w = (img.shape[1] - 1) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for oi, i in enumerate(range(0, img.shape[0], stride)):
        for oj, j in enumerate(range(0, img.shape[1], stride)):
            region = padded[i:i + kh, j:j + kw]
            out[oi, oj] = np.sum(region * kernel)
    return out


KERNELS = {
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "blur": np.ones((3, 3), dtype=np.float32) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
}


def section16_convolution(data):
    print("\n=== BÖLÜM 16: Konvolüsyon Katmanı — Sıfırdan NumPy ===")
    sample = data["X_train"][int(np.argmax(data["y_train"]))]  # bir pozitif (plaka) örneği seç

    fig, axes = plt.subplots(1, len(KERNELS) + 1, figsize=(3 * (len(KERNELS) + 1), 3))
    axes[0].imshow(sample, cmap="gray"); axes[0].set_title("Orijinal"); axes[0].axis("off")
    feature_maps = {}
    for ax, (name, kernel) in zip(axes[1:], KERNELS.items()):
        fmap = conv2d(sample, kernel)
        feature_maps[name] = fmap
        ax.imshow(fmap, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    plt.suptitle("Sıfırdan Konvolüsyon — Manuel Çekirdekler")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "16_konvolusyon_ornekleri.png"), dpi=120)
    plt.close()

    print("Konvolüsyon çıktıları kaydedildi.")
    return sample, feature_maps


# ==========================================================================
# BÖLÜM 17 — AKTİVASYON FONKSİYONLARI VE ÖZELLİK HARİTALARI
# ==========================================================================

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh_act(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


ACTIVATIONS = {"ReLU": relu, "Sigmoid": sigmoid, "Tanh": tanh_act, "LeakyReLU": leaky_relu}


def section17_activations(sample, feature_maps):
    print("\n=== BÖLÜM 17: Aktivasyon Fonksiyonları ===")

    # Aktivasyon eğrileri
    x = np.linspace(-5, 5, 200)
    plt.figure(figsize=(6, 4))
    for name, fn in ACTIVATIONS.items():
        plt.plot(x, fn(x), label=name)
    plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
    plt.legend(); plt.title("Aktivasyon Fonksiyonları")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "17_aktivasyon_egrileri.png"), dpi=120)
    plt.close()

    # sobel_x özellik haritasına aktivasyon uygulanmış hali
    fmap = feature_maps["sobel_x"]
    fig, axes = plt.subplots(1, len(ACTIVATIONS) + 1, figsize=(3 * (len(ACTIVATIONS) + 1), 3))
    axes[0].imshow(fmap, cmap="gray"); axes[0].set_title("Ham Özellik Haritası (sobel_x)"); axes[0].axis("off")
    for ax, (name, fn) in zip(axes[1:], ACTIVATIONS.items()):
        ax.imshow(fn(fmap), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    plt.suptitle("Aktivasyon Sonrası Özellik Haritaları")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "17_aktivasyon_sonrasi_haritalar.png"), dpi=120)
    plt.close()

    print("Aktivasyon görselleri kaydedildi.")


# ==========================================================================
# BÖLÜM 18 — TAM BAĞLI SİNİR AĞI (MLP) İLE SINIFLANDIRMA
# ==========================================================================

def section18_mlp(data):
    print("\n=== BÖLÜM 18: MLP ile Sınıflandırma ===")
    X_train = data["X_train_aug"].reshape(len(data["X_train_aug"]), -1)
    y_train = data["y_train_aug"]
    X_valid = data["X_valid"].reshape(len(data["X_valid"]), -1)
    y_valid = data["y_valid"]
    X_test = data["X_test"].reshape(len(data["X_test"]), -1)
    y_test = data["y_test"]

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train, y_train)

    for split_name, X_s, y_s in [("Valid", X_valid, y_valid), ("Test", X_test, y_test)]:
        if len(X_s) == 0:
            continue
        y_pred = mlp.predict(X_s)
        acc = accuracy_score(y_s, y_pred)
        print(f"MLP {split_name} doğruluk: {acc:.3f}")

    if len(X_test) > 0:
        y_pred_test = mlp.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(cm, display_labels=["plaka-değil", "plaka"])
        disp.plot(cmap="Blues")
        plt.title("MLP — Test Seti Karışıklık Matrisi")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "18_mlp_confusion_matrix.png"), dpi=120)
        plt.close()

    return mlp


# ==========================================================================
# BÖLÜM 19 — SVM vs MLP vs KNN — KAPSAMLI KARŞILAŞTIRMA
# ==========================================================================

def section19_compare_models(data, mlp):
    print("\n=== BÖLÜM 19: SVM vs MLP vs KNN Karşılaştırma ===")
    X_train = data["X_train_aug"].reshape(len(data["X_train_aug"]), -1)
    y_train = data["y_train_aug"]
    X_test = data["X_test"].reshape(len(data["X_test"]), -1)
    y_test = data["y_test"]

    if len(X_test) == 0:
        print("Test seti boş, karşılaştırma atlanıyor.")
        return {}

    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    results = {}
    for name, model in [("SVM", svm), ("MLP", mlp), ("KNN", knn)]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"{name}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")

    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.25
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(results):
        vals = [results[name][m] for m in metrics]
        plt.bar(x + i * width, vals, width, label=name)
    plt.xticks(x + width, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Skor")
    plt.title("SVM vs MLP vs KNN — Test Seti Karşılaştırması")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "19_model_karsilastirma.png"), dpi=120)
    plt.close()

    return {"svm": svm, "knn": knn, "results": results}


# ==========================================================================
# BÖLÜM 20 — GRAD-CAM BENZERİ ISI HARİTASI (OCCLUSION SENSITIVITY)
# ==========================================================================

def occlusion_saliency(model, img, patch_size=6, stride=3):
    """
    Gerçek Grad-CAM, eğitilmiş bir CNN'in gradyanlarına ihtiyaç duyar; burada
    framework bağımsız (numpy + sklearn modeliyle çalışan) bir alternatif olarak
    "occlusion sensitivity" kullanılıyor: görüntü üzerinde küçük bir yama
    kaydırılıp örtülüyor, modelin "plaka" olasılığındaki düşüş ısı haritasına
    yazılıyor. Yüksek düşüş = model o bölgeye çok güveniyor demek.
    """
    h, w = img.shape
    base_prob = model.predict_proba(img.reshape(1, -1))[0, 1]
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            occluded = img.copy()
            occluded[i:i + patch_size, j:j + patch_size] = 0.5
            prob = model.predict_proba(occluded.reshape(1, -1))[0, 1]
            drop = max(0.0, base_prob - prob)
            heatmap[i:i + patch_size, j:j + patch_size] += drop
            counts[i:i + patch_size, j:j + patch_size] += 1

    counts[counts == 0] = 1
    heatmap = heatmap / counts
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap, base_prob


def section20_saliency(data, mlp):
    print("\n=== BÖLÜM 20: Grad-CAM Benzeri Isı Haritası ===")
    pos_idx = np.where(data["y_test"] == 1)[0]
    if len(pos_idx) == 0:
        pos_idx = np.where(data["y_train"] == 1)[0]
        sample_img = data["X_train"][pos_idx[0]] if len(pos_idx) else data["X_train"][0]
    else:
        sample_img = data["X_test"][pos_idx[0]]

    heatmap, base_prob = occlusion_saliency(mlp, sample_img)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(sample_img, cmap="gray"); axes[0].set_title("Orijinal"); axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Isı Haritası"); axes[1].axis("off")
    axes[2].imshow(sample_img, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay (p={base_prob:.2f})")
    axes[2].axis("off")
    plt.suptitle("Occlusion Sensitivity — Grad-CAM Benzeri Görselleştirme")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "20_saliency_heatmap.png"), dpi=120)
    plt.close()

    print(f"Isı haritası kaydedildi. Modelin bu örnek için 'plaka' olasılığı: {base_prob:.3f}")


# ==========================================================================
# BÖLÜM 21 — GERÇEK CNN PIPELINE — SIFIRDAN İLERİ YAYILIM
# ==========================================================================

def max_pool2d(fmap, pool_size=2, stride=2):
    h, w = fmap.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            region = fmap[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i, j] = np.max(region)
    return out


def cnn_forward_pass(img, kernels):
    """
    Basit bir CNN'in ileri yayılımını sıfırdan gösterir:
    Conv -> ReLU -> MaxPool -> Flatten -> Dense (rastgele ağırlıklarla, gösterim amaçlı)
    """
    conv_outputs = [conv2d(img, k) for k in kernels.values()]
    relu_outputs = [relu(c) for c in conv_outputs]
    pooled_outputs = [max_pool2d(r) for r in relu_outputs]

    flattened = np.concatenate([p.flatten() for p in pooled_outputs])
    rng = np.random.RandomState(RANDOM_STATE)
    W_dense = rng.randn(flattened.shape[0], 2) * 0.01
    b_dense = np.zeros(2)
    logits = flattened @ W_dense + b_dense
    probs = np.exp(logits) / np.sum(np.exp(logits))

    return {
        "conv": conv_outputs,
        "relu": relu_outputs,
        "pooled": pooled_outputs,
        "flattened": flattened,
        "probs": probs,
    }


def section21_cnn_pipeline(data):
    print("\n=== BÖLÜM 21: Gerçek CNN Pipeline — Sıfırdan İleri Yayılım ===")
    pos_idx = np.where(data["y_train"] == 1)[0]
    sample = data["X_train"][pos_idx[0]] if len(pos_idx) else data["X_train"][0]

    result = cnn_forward_pass(sample, KERNELS)

    n_k = len(KERNELS)
    fig, axes = plt.subplots(3, n_k, figsize=(3 * n_k, 8))
    kernel_names = list(KERNELS.keys())
    for j, name in enumerate(kernel_names):
        axes[0, j].imshow(result["conv"][j], cmap="gray"); axes[0, j].set_title(f"Conv: {name}"); axes[0, j].axis("off")
        axes[1, j].imshow(result["relu"][j], cmap="gray"); axes[1, j].set_title("ReLU"); axes[1, j].axis("off")
        axes[2, j].imshow(result["pooled"][j], cmap="gray"); axes[2, j].set_title("MaxPool"); axes[2, j].axis("off")
    plt.suptitle("Sıfırdan CNN İleri Yayılım: Conv -> ReLU -> MaxPool")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "21_cnn_pipeline_ileri_yayilim.png"), dpi=120)
    plt.close()

    print(f"Flatten boyutu: {result['flattened'].shape[0]}")
    print(f"(Eğitilmemiş, rastgele ağırlıklı) çıkış olasılıkları: {result['probs']}")


# ==========================================================================
# ANA AKIŞ
# ==========================================================================

def main():
    data = section13_prepare_and_augment()
    pca, X_pca = section14_pca(data)
    section15_tsne(data, pca)
    sample, feature_maps = section16_convolution(data)
    section17_activations(sample, feature_maps)
    mlp = section18_mlp(data)
    section19_compare_models(data, mlp)
    section20_saliency(data, mlp)
    section21_cnn_pipeline(data)
    print(f"\nTüm görseller '{OUT_DIR}/' klasörüne kaydedildi.")
    data = section13_prepare_and_augment()
    # ... diğer bölümler ...
    mlp = section18_mlp(data)
    # ... diğer bölümler ...
    print(f"\nTüm görseller '{OUT_DIR}/' klasörüne kaydedildi.")
    return mlp  # <-- Bu satırın var olduğundan emin olun!

if __name__ == "__main__":
    main()



def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"\n[!] Hata: {image_path} bulunamadı!")
        return
    
    # Eğitimdeki gibi ön işleme
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    input_data = img_norm.reshape(1, -1)
    
    # Tahmin
    prob = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]
    
    label = "PLAKA" if prediction == 1 else "PLAKA DEĞİL"
    print(f"\n--- Tahmin Sonucu ---")
    print(f"Görsel: {image_path}")
    print(f"Tahmin: {label}")
    print(f"Güven Skoru: Plaka için %{prob[1]*100:.2f}")

if __name__ == "__main__":
    # Önce pipeline'ı çalıştırıp eğitilmiş modeli alıyoruz
    egitilmis_model = main()
    
    # Test etmek istediğin görselin yolu (dosyayı proje dizinine koy)
    test_resmi = "/Users/d1-19/Desktop/görüntü-işleme/data/test/car-81-_jpg.rf.d0315fae2dadf61df1e5c6d712912210.jpg"
    
    if os.path.exists(test_resmi):
        predict_image(test_resmi, egitilmis_model)
    else:
        print(f"\nNot: Tahmin yapmak için '{test_resmi}' dosyasını ekleyebilirsin.")

