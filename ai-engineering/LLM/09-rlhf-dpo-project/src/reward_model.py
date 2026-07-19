"""
reward_model.py
----------------
RLHF pipeline'ının 2. adımı: bir ÖDÜL MODELİ (Reward Model, RM) eğitimi.

Gerçek RLHF'de bu model genelde SFT modelinin bir kopyası olup son katmanı
skaler bir ödül çıktısı verecek şekilde değiştirilmiş bir sinir ağıdır.
Burada, konuyu koddan takip etmeyi kolaylaştırmak için ödül fonksiyonu
tablo (lookup table) şeklinde parametrelendirilmiştir: her (prompt, cevap)
çifti için doğrudan bir r_phi[p, k] skaları öğrenilir. Matematik ve eğitim
dinamiği, nöral bir ağla birebir aynıdır — sadece fonksiyon sınıfı basitleştirilmiştir.

Kayıp fonksiyonu, Bradley-Terry modelinin negatif log-olabilirliğidir:

    L(phi) = - E_{(p,i,j) ~ tercihler} [ log sigmoid( r_phi[p,i] - r_phi[p,j] ) ]

Burada i, insan tarafından j'ye tercih edilen cevaptır.
"""

import numpy as np
from environment import sigmoid


def train_reward_model(preferences, n_prompts, k_responses,
                        lr=0.5, epochs=300, seed=0):
    """
    Basit tam-batch gradyan inişi ile ödül modeli eğitir.

    Gradyan türetimi:
        d = r_i - r_j
        loss = -log(sigmoid(d)) = softplus(-d)
        dloss/dd = sigmoid(d) - 1

        dloss/dr_i =  (sigmoid(d) - 1)
        dloss/dr_j = -(sigmoid(d) - 1)

    Yani model "i, j'den daha kaliteli" tahminini ne kadar az güvenle
    yapıyorsa (sigmoid(d) 1'den ne kadar uzaksa), güncelleme o kadar büyük olur.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, size=(n_prompts, k_responses))

    loss_history = []

    for epoch in range(epochs):
        grad = np.zeros_like(r)
        total_loss = 0.0

        for (p, i, j) in preferences:
            d = r[p, i] - r[p, j]
            sig = sigmoid(d)
            total_loss += -np.log(sig + 1e-12)

            g = sig - 1.0          # dloss/dr_i
            grad[p, i] += g
            grad[p, j] += -g

        n = len(preferences)
        grad /= n
        r -= lr * grad

        loss_history.append(total_loss / n)

    return r, loss_history


def reward_model_accuracy(r_phi, preferences):
    """Ödül modelinin, tercih verisindeki 'kim kazandı' etiketlerini
    ne oranda doğru tahmin ettiğini ölçer (eğitim verisi üzerinde)."""
    correct = 0
    for (p, i, j) in preferences:
        if r_phi[p, i] > r_phi[p, j]:
            correct += 1
    return correct / len(preferences)
