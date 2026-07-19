"""
dpo.py
------
DPO (Direct Preference Optimization) — Rafailov ve ark., 2023.

RLHF'in "ödül modeli eğit -> RL ile politikayı optimize et" şeklindeki iki
aşamalı, kararsız ve pahalı sürecini TEK bir denetimli öğrenme kaybına
indirger. Kilit fikir: rlhf.py'deki kapalı-form çözümü

    pi*(k) ∝ pi_ref(k) * exp( r(k) / beta )

tersine çevrilebilir:

    r(k) = beta * log( pi*(k) / pi_ref(k) )  + sabit

Yani "gizli/örtük" bir ödül fonksiyonu, doğrudan politikanın kendisinden
(referans politikaya oranla) ifade edilebilir. Bu ödül ifadesi, Bradley-Terry
tercih kaybının içine yerleştirildiğinde, ayrı bir ödül modeline hiç gerek
kalmadan politikayı doğrudan tercih verisiyle eğitebileceğimiz DPO kaybı
ortaya çıkar:

    L_DPO(theta) = - E_{(p,i,j)} [
        log sigmoid( beta * [ (log pi_theta(i) - log pi_ref(i))
                             - (log pi_theta(j) - log pi_ref(j)) ] )
    ]

Burada i tercih edilen (chosen), j reddedilen (rejected) cevaptır.
Bu kayıp, "chosen cevabın referansa göre olasılığını artır, rejected
cevabınkini azalt — ama ne kadar agresif olacağını beta ve mevcut
modelin ne kadar yanıldığı belirlesin" şeklinde yorumlanabilir.
"""

import numpy as np
from rlhf import softmax


def dpo_loss_and_grad(logits, pi_ref, preferences, beta):
    """
    Tüm tercih veri kümesi için DPO kaybını ve logits'e göre gradyanını
    (tam-batch) hesaplar.

    Gradyan türetimi:
        u = beta * [ (log pi_i - log piref_i) - (log pi_j - log piref_j) ]
        loss = -log sigmoid(u) = softplus(-u)
        dloss/du = sigmoid(u) - 1

        d(log pi_k)/dlogits_m = delta(k,m) - pi_m   (softmax log-olabilirlik türevi)

        d(loss)/dlogits_m
            = beta*(sigmoid(u)-1) * [delta(i,m) - pi_m] - beta*(sigmoid(u)-1) * [delta(j,m) - pi_m]
            = beta*(sigmoid(u)-1) * [delta(i,m) - delta(j,m)]
            (pi_m terimleri birbirini götürür — çok temiz bir sonuç!)

    Yani pratikte: her (p, i, j) çifti için sadece i ve j indekslerine
    +/- bir miktar gradyan eklenir; softmax'ın geri kalanına dokunmaya
    gerek yoktur.
    """
    pi = softmax(logits, axis=-1)
    log_pi = np.log(pi + 1e-12)
    log_pi_ref = np.log(pi_ref + 1e-12)

    grad = np.zeros_like(logits)
    total_loss = 0.0

    for (p, i, j) in preferences:
        u = beta * ((log_pi[p, i] - log_pi_ref[p, i]) - (log_pi[p, j] - log_pi_ref[p, j]))
        sig = 1.0 / (1.0 + np.exp(-u))
        total_loss += -np.log(sig + 1e-12)

        coef = beta * (sig - 1.0)
        grad[p, i] += coef
        grad[p, j] += -coef

    n = len(preferences)
    return total_loss / n, grad / n


def train_dpo_policy(pi_ref, preferences, beta=0.5, lr=0.3, epochs=300):
    """DPO kaybını tam-batch gradyan inişi ile optimize eder."""
    n_prompts, k_responses = pi_ref.shape
    logits = np.zeros((n_prompts, k_responses))  # referans (SFT) politikasıyla başla

    loss_history = []

    for epoch in range(epochs):
        loss, grad = dpo_loss_and_grad(logits, pi_ref, preferences, beta)
        loss_history.append(loss)
        logits -= lr * grad   # kayıp minimize edildiği için gradyan İNİŞİ

    final_pi = softmax(logits, axis=-1)
    return final_pi, logits, loss_history


def implicit_reward(logits, pi_ref, beta):
    """
    DPO ile eğitilmiş bir politikadan, "örtük ödül fonksiyonunu" geri
    çıkarır:  r(k) = beta * log( pi_theta(k) / pi_ref(k) )

    Bu, RLHF pipeline'ındaki reward_model.py ile eğitilen r_phi'ye
    KAVRAMSAL OLARAK KARŞILIK GELİR — DPO bunu hiç ayrı bir model eğitmeden,
    politikanın kendisinden elde eder.
    """
    pi = softmax(logits, axis=-1)
    return beta * (np.log(pi + 1e-12) - np.log(pi_ref + 1e-12))
