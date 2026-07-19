"""
quantizer.py
------------
Kuantizasyon (quantization) işlemlerinin temel matematiğini uygulayan modül.

Bu modül, float32 değerleri düşük bit'li tam sayılara (INT8 gibi) dönüştüren
(quantize) ve geri dönüştüren (dequantize) fonksiyonları içerir.

İki temel yöntem uygulanmıştır:
  1) Simetrik Kuantizasyon (Symmetric Quantization)
  2) Asimetrik Kuantizasyon (Asymmetric Quantization)

Her ikisi de "uniform affine quantization" ailesindendir:
    q = round(x / scale) + zero_point
    x_hat = (q - zero_point) * scale
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class QuantizationParams:
    """Bir tensörü kuantize etmek için gereken parametreler."""
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    mode: str  # "symmetric" | "asymmetric"


def compute_symmetric_params(x: np.ndarray, num_bits: int = 8) -> QuantizationParams:
    """
    Simetrik kuantizasyon parametrelerini hesaplar.

    Aralık [-max(|x|), +max(|x|)] etrafında simetrik kabul edilir.
    zero_point her zaman 0'dır çünkü sıfır, sıfıra tam olarak eşlenir.
    Ağırlıklar (weights) genelde sıfır etrafında simetrik dağıldığı için
    bu yöntem ağırlık kuantizasyonunda yaygın kullanılır.
    """
    qmax = 2 ** (num_bits - 1) - 1   # INT8 için: 127
    qmin = -(2 ** (num_bits - 1))    # INT8 için: -128

    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        max_abs = 1e-8  # sıfıra bölmeyi önle

    scale = max_abs / qmax
    zero_point = 0

    return QuantizationParams(scale=scale, zero_point=zero_point,
                               qmin=qmin, qmax=qmax, mode="symmetric")


def compute_asymmetric_params(x: np.ndarray, num_bits: int = 8) -> QuantizationParams:
    """
    Asimetrik kuantizasyon parametrelerini hesaplar.

    Aralık [min(x), max(x)] olarak alınır ve bu aralık [qmin, qmax]
    tam sayı aralığına eşlenir. zero_point genelde 0 değildir.
    ReLU sonrası aktivasyonlar gibi tek taraflı (hep pozitif) dağılımlarda
    simetrik yönteme göre daha az bilgi kaybı sağlar.
    """
    qmax = 2 ** num_bits - 1  # UINT8 için: 255
    qmin = 0

    x_min = np.min(x)
    x_max = np.max(x)

    # Aralık sıfır genişliğinde olmasın
    if x_max == x_min:
        x_max = x_min + 1e-8

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = round(qmin - x_min / scale)
    zero_point = int(np.clip(zero_point, qmin, qmax))

    return QuantizationParams(scale=scale, zero_point=zero_point,
                               qmin=qmin, qmax=qmax, mode="asymmetric")


def quantize(x: np.ndarray, params: QuantizationParams) -> np.ndarray:
    """
    Float32 tensörü, verilen parametrelerle tam sayıya (INT8/UINT8) çevirir.

        q = clip(round(x / scale) + zero_point, qmin, qmax)
    """
    q = np.round(x / params.scale) + params.zero_point
    q = np.clip(q, params.qmin, params.qmax)
    dtype = np.int8 if params.mode == "symmetric" else np.uint8
    return q.astype(dtype)


def dequantize(q: np.ndarray, params: QuantizationParams) -> np.ndarray:
    """
    Kuantize edilmiş tam sayı tensörünü tekrar float32'ye çevirir (yaklaşık).

        x_hat = (q - zero_point) * scale
    """
    return (q.astype(np.float32) - params.zero_point) * params.scale


def quantization_error_report(x: np.ndarray, x_hat: np.ndarray) -> dict:
    """Orijinal ve kuantize-sonrası-geri-dönüştürülmüş değerler arasındaki hatayı ölçer."""
    diff = x.astype(np.float32) - x_hat.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    # Sinyal-Gürültü Oranı (dB) - kuantizasyon gürültüsünün büyüklüğünü ölçer
    signal_power = float(np.mean(x.astype(np.float32) ** 2))
    sqnr_db = 10 * np.log10(signal_power / mse) if mse > 0 else float("inf")

    return {
        "mse": mse,
        "mae": mae,
        "max_abs_error": max_err,
        "sqnr_db": sqnr_db,
    }


def memory_footprint_bytes(x: np.ndarray, num_bits: int) -> int:
    """Verilen bit genişliğiyle bir tensörün kapladığı bellek miktarını (byte) hesaplar."""
    return int(x.size * num_bits / 8)
