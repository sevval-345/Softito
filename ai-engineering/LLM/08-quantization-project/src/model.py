"""
model.py
--------
Kuantizasyonun gerçek bir modelde nasıl bir etki yarattığını göstermek için
NumPy ile yazılmış çok basit bir 2 katmanlı MLP (Multi-Layer Perceptron).

Amaç karmaşık bir model eğitmek değil; rastgele (ama sabit seed'li) ağırlıklarla
ileri besleme (forward pass) yapıp, ağırlıklar kuantize edildiğinde çıktının
ne kadar değiştiğini gözlemlemektir.
"""

import numpy as np


class TinyMLP:
    """Girdi -> Gizli katman (ReLU) -> Çıktı katmanı şeklinde basit bir ağ."""

    def __init__(self, in_dim=32, hidden_dim=64, out_dim=10, seed=42):
        rng = np.random.default_rng(seed)
        # Xavier benzeri ölçekleme ile gerçekçi ağırlık dağılımı
        self.W1 = rng.normal(0, 1 / np.sqrt(in_dim), size=(in_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden_dim), size=(hidden_dim, out_dim)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray, W1=None, W2=None) -> np.ndarray:
        """W1/W2 verilmezse orijinal (float32) ağırlıklar kullanılır.
        Kuantize edilmiş ağırlıklarla test etmek için W1/W2 override edilebilir."""
        W1 = self.W1 if W1 is None else W1
        W2 = self.W2 if W2 is None else W2

        h = np.maximum(0, x @ W1 + self.b1)  # ReLU
        out = h @ W2 + self.b2
        return out

    def all_weights(self):
        return {"W1": self.W1, "W2": self.W2}
