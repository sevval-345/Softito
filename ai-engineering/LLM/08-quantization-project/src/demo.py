"""
demo.py
-------
Projenin ana çalıştırılabilir dosyası.

Bu script:
  1) Rastgele bir ağırlık tensörü ve basit bir MLP modeli oluşturur.
  2) Ağırlıkları hem SİMETRİK hem ASİMETRİK INT8 kuantizasyon ile kuantize eder.
  3) Kuantizasyon hatasını (MSE, MAE, SQNR) ve bellek tasarrufunu raporlar.
  4) Kuantize edilmiş ağırlıklarla modeli çalıştırıp çıktıdaki sapmayı gösterir.
  5) Sonuçları outputs/ klasörüne bir grafik ve bir metin raporu olarak kaydeder.

Çalıştırma:
    python src/demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantizer import (
    compute_symmetric_params,
    compute_asymmetric_params,
    quantize,
    dequantize,
    quantization_error_report,
    memory_footprint_bytes,
)
from model import TinyMLP

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def separator(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_weight_quantization_demo():
    separator("1) AĞIRLIK TENSÖRÜ ÜZERİNDE KUANTİZASYON")

    rng = np.random.default_rng(7)
    # Gerçekçi bir ağırlık dağılımı: sıfır etrafında, çan eğrisi (normal dağılım)
    weights = rng.normal(loc=0.0, scale=0.5, size=(256, 256)).astype(np.float32)

    results = {}

    for mode, param_fn in [
        ("symmetric", compute_symmetric_params),
        ("asymmetric", compute_asymmetric_params),
    ]:
        params = param_fn(weights, num_bits=8)
        q = quantize(weights, params)
        deq = dequantize(q, params)
        err = quantization_error_report(weights, deq)

        fp32_bytes = memory_footprint_bytes(weights, 32)
        int8_bytes = memory_footprint_bytes(weights, 8)

        results[mode] = {
            "params": params,
            "quantized": q,
            "dequantized": deq,
            "error": err,
            "fp32_bytes": fp32_bytes,
            "int8_bytes": int8_bytes,
        }

        print(f"\n--- {mode.upper()} KUANTİZASYON ---")
        print(f"  scale       : {params.scale:.6f}")
        print(f"  zero_point  : {params.zero_point}")
        print(f"  aralık      : [{params.qmin}, {params.qmax}]")
        print(f"  MSE         : {err['mse']:.8f}")
        print(f"  MAE         : {err['mae']:.6f}")
        print(f"  Max hata    : {err['max_abs_error']:.6f}")
        print(f"  SQNR        : {err['sqnr_db']:.2f} dB  (yüksek = daha iyi)")
        print(f"  FP32 bellek : {fp32_bytes:,} byte")
        print(f"  INT8 bellek : {int8_bytes:,} byte")
        print(f"  Sıkıştırma  : {fp32_bytes / int8_bytes:.2f}x")

    return weights, results


def run_model_inference_demo(weights_report):
    separator("2) BASİT BİR MODEL ÜZERİNDE UÇTAN UCA ETKİ")

    model = TinyMLP(in_dim=32, hidden_dim=64, out_dim=10, seed=42)
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, size=(8, 32)).astype(np.float32)  # 8 örnekli mini batch

    # 1) Orijinal FP32 çıktısı
    out_fp32 = model.forward(x)

    # 2) Ağırlıkları simetrik INT8 ile kuantize edip geri çöz (dequantize),
    #    ardından bu "sahte kuantize" (fake-quantized) ağırlıklarla forward yap.
    #    Bu, gerçek INT8 donanımının davranışını float alanında simüle eder.
    p1 = compute_symmetric_params(model.W1, num_bits=8)
    p2 = compute_symmetric_params(model.W2, num_bits=8)
    W1_q = dequantize(quantize(model.W1, p1), p1)
    W2_q = dequantize(quantize(model.W2, p2), p2)

    out_int8 = model.forward(x, W1=W1_q, W2=W2_q)

    diff = out_fp32 - out_int8
    rel_err = np.linalg.norm(diff) / np.linalg.norm(out_fp32)

    print(f"\nÇıktı boyutu           : {out_fp32.shape}")
    print(f"Ortalama mutlak fark   : {np.mean(np.abs(diff)):.6f}")
    print(f"Göreli L2 hata         : {rel_err * 100:.3f}%")
    print("\nÖrnek karşılaştırma (ilk örnek, ilk 5 çıktı nöronu):")
    print(f"  FP32 : {np.round(out_fp32[0][:5], 4)}")
    print(f"  INT8 : {np.round(out_int8[0][:5], 4)}")

    return out_fp32, out_int8


def plot_results(weights, results):
    """Orijinal ve kuantize edilmiş ağırlık dağılımlarını görselleştirir."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].hist(weights.flatten(), bins=80, color="#4C72B0")
    axes[0].set_title("Orijinal FP32 Ağırlıklar")
    axes[0].set_xlabel("Değer")
    axes[0].set_ylabel("Frekans")

    axes[1].hist(results["symmetric"]["dequantized"].flatten(), bins=80, color="#55A868")
    axes[1].set_title("Simetrik INT8 (Dequantize Sonrası)")
    axes[1].set_xlabel("Değer")

    axes[2].hist(results["asymmetric"]["dequantized"].flatten(), bins=80, color="#C44E52")
    axes[2].set_title("Asimetrik UINT8 (Dequantize Sonrası)")
    axes[2].set_xlabel("Değer")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.suptitle("Kuantizasyon Öncesi ve Sonrası Ağırlık Dağılımları", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "quantization_histograms.png")
    fig.savefig(out_path, dpi=150)
    print(f"\n[Grafik kaydedildi] {out_path}")

    # İkinci grafik: bit genişliğine göre bellek/hata trade-off'u
    bit_widths = [2, 4, 6, 8, 16]
    mses = []
    compressions = []
    for b in bit_widths:
        p = compute_symmetric_params(weights, num_bits=b)
        deq = dequantize(quantize(weights, p), p)
        err = quantization_error_report(weights, deq)
        mses.append(err["mse"])
        compressions.append(32 / b)

    fig2, ax1 = plt.subplots(figsize=(7, 4.5))
    color1 = "#4C72B0"
    ax1.plot(bit_widths, mses, marker="o", color=color1, label="MSE")
    ax1.set_xlabel("Bit Genişliği (num_bits)")
    ax1.set_ylabel("MSE (hata)", color=color1)
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#C44E52"
    ax2.plot(bit_widths, compressions, marker="s", color=color2, label="Sıkıştırma Oranı")
    ax2.set_ylabel("Sıkıştırma Oranı (x)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig2.suptitle("Bit Genişliği: Hata vs. Sıkıştırma Trade-off'u")
    fig2.tight_layout()

    out_path2 = os.path.join(OUTPUT_DIR, "bitwidth_tradeoff.png")
    fig2.savefig(out_path2, dpi=150)
    print(f"[Grafik kaydedildi] {out_path2}")


def write_text_report(weights, results, out_fp32, out_int8):
    lines = []
    lines.append("KUANTİZASYON DEMO RAPORU")
    lines.append("=" * 40)
    for mode in ["symmetric", "asymmetric"]:
        r = results[mode]
        lines.append(f"\n[{mode.upper()}]")
        lines.append(f"  scale={r['params'].scale:.6f}  zero_point={r['params'].zero_point}")
        lines.append(f"  MSE={r['error']['mse']:.8f}  SQNR={r['error']['sqnr_db']:.2f} dB")
        lines.append(f"  Bellek: {r['fp32_bytes']:,} B -> {r['int8_bytes']:,} B "
                      f"({r['fp32_bytes'] / r['int8_bytes']:.2f}x sıkıştırma)")

    rel_err = np.linalg.norm(out_fp32 - out_int8) / np.linalg.norm(out_fp32)
    lines.append(f"\n[MODEL ÇIKTISI ÜZERİNDE ETKİ]")
    lines.append(f"  Göreli L2 hata: {rel_err * 100:.3f}%")

    out_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Rapor kaydedildi] {out_path}")


def main():
    weights, results = run_weight_quantization_demo()
    out_fp32, out_int8 = run_model_inference_demo(results)
    plot_results(weights, results)
    write_text_report(weights, results, out_fp32, out_int8)
    separator("TAMAMLANDI")
    print(f"Tüm çıktılar '{os.path.abspath(OUTPUT_DIR)}' klasöründe.")


if __name__ == "__main__":
    main()
