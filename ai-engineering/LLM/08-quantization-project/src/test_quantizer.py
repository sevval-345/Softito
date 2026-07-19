"""
test_quantizer.py
------------------
quantizer.py için basit doğrulama testleri.
Çalıştırma: python -m pytest src/test_quantizer.py -v   (pytest varsa)
            veya doğrudan:  python src/test_quantizer.py
"""

import numpy as np
from quantizer import (
    compute_symmetric_params,
    compute_asymmetric_params,
    quantize,
    dequantize,
    quantization_error_report,
)


def test_symmetric_round_trip_small_error():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    params = compute_symmetric_params(x, num_bits=8)
    q = quantize(x, params)
    x_hat = dequantize(q, params)
    err = quantization_error_report(x, x_hat)
    assert err["max_abs_error"] < 0.02, "Simetrik kuantizasyon hatası beklenenden büyük"


def test_asymmetric_handles_positive_only_range():
    x = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    params = compute_asymmetric_params(x, num_bits=8)
    q = quantize(x, params)
    x_hat = dequantize(q, params)
    assert np.allclose(x_hat, x, atol=0.05)


def test_zero_maps_to_zero_in_symmetric_mode():
    x = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
    params = compute_symmetric_params(x, num_bits=8)
    q = quantize(np.array([0.0], dtype=np.float32), params)
    assert q[0] == 0, "Simetrik modda sıfır her zaman sıfıra eşlenmeli"


def test_quantized_dtype_ranges():
    x = np.random.default_rng(0).normal(0, 1, size=1000).astype(np.float32)
    p_sym = compute_symmetric_params(x, num_bits=8)
    q_sym = quantize(x, p_sym)
    assert q_sym.min() >= -128 and q_sym.max() <= 127

    p_asym = compute_asymmetric_params(x, num_bits=8)
    q_asym = quantize(x, p_asym)
    assert q_asym.min() >= 0 and q_asym.max() <= 255


def _run_all():
    tests = [
        test_symmetric_round_trip_small_error,
        test_asymmetric_handles_positive_only_range,
        test_zero_maps_to_zero_in_symmetric_mode,
        test_quantized_dtype_ranges,
    ]
    for t in tests:
        t()
        print(f"OK  - {t.__name__}")
    print("\nTüm testler başarıyla geçti.")


if __name__ == "__main__":
    _run_all()
