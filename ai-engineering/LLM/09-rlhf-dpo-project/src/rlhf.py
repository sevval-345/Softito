"""
rlhf.py
-------
RLHF pipeline'ının 3. (ve en zor) adımı: öğrenilen ödül modelini kullanarak
politikayı (dil modelini) pekiştirmeli öğrenme ile güncellemek.

Gerçek RLHF sistemlerinde bu adım PPO (Proximal Policy Optimization) ile,
örnekleme (sampling) yoluyla yapılır — pahalı, yüksek varyanslı ve kararsız
olabilen bir süreçtir. Burada cevap uzayı küçük ve ayrık olduğu için
(K_RESPONSES adet), beklentiyi (expectation) TAM olarak hesaplayabiliyoruz;
bu da örnekleme gürültüsü olmadan, PPO'nun optimize ettiği AYNI amaç
fonksiyonunu gradyan çıkışı (gradient ascent) ile doğrudan optimize etmemizi
sağlıyor. Yani bu, "sonsuz örneklemli, sıfır varyanslı PPO" gibi düşünülebilir.

Amaç fonksiyonu (her prompt için, KL-düzenlemeli ödül maksimizasyonu):

    J(theta) = E_{a ~ pi_theta}[ r_phi(a) ]  -  beta * KL( pi_theta || pi_ref )

    KL(pi || pi_ref) = sum_k pi_k * log(pi_k / pi_ref_k)

Bu ifadedeki beta * KL cezası, PPO'da kullanılan "clip" mekanizmasıyla aynı
amaca hizmet eder: politikanın referans (SFT) modelinden çok fazla
uzaklaşmasını, dolayısıyla ödül modelini kandırmaya çalışmasını
(reward hacking) engellemek.

Gradyan türetimi (softmax parametrelendirmesi ile):
    pi_k = softmax(logits)_k

    dJ/dpi_k = r_phi[k] - beta * (log(pi_k / pi_ref_k) + 1)

    dJ/dlogits_m = pi_m * ( dJ/dpi_m - sum_k pi_k * dJ/dpi_k )
                   (softmax Jacobian-vector çarpımının standart formülü)
"""

import numpy as np


def softmax(logits, axis=-1):
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def rlhf_policy_gradient_step(logits, r_phi, pi_ref, beta, lr):
    """Tek bir gradyan çıkışı adımı uygular ve güncellenmiş logits'i döndürür."""
    pi = softmax(logits, axis=-1)

    # dJ/dpi_k
    dJ_dpi = r_phi - beta * (np.log(pi + 1e-12) - np.log(pi_ref + 1e-12) + 1.0)

    # softmax Jacobian-vector çarpımı: dJ/dlogits = pi * (dJ/dpi - sum(pi * dJ/dpi))
    weighted_sum = np.sum(pi * dJ_dpi, axis=-1, keepdims=True)
    dJ_dlogits = pi * (dJ_dpi - weighted_sum)

    # Gradyan ÇIKIŞI (ascent) yapıyoruz çünkü J'yi maksimize ediyoruz
    new_logits = logits + lr * dJ_dlogits
    return new_logits


def train_rlhf_policy(r_phi, pi_ref, beta=0.5, lr=0.3, epochs=300):
    """
    KL-düzenlemeli politika optimizasyonunu çalıştırır ve eğitim boyunca
    beklenen ödül / KL geçmişini döndürür.
    """
    n_prompts, k_responses = r_phi.shape
    logits = np.zeros((n_prompts, k_responses))  # uniform politika ile başla (SFT gibi)

    reward_history = []
    kl_history = []

    for epoch in range(epochs):
        pi = softmax(logits, axis=-1)

        expected_reward = float(np.mean(np.sum(pi * r_phi, axis=1)))
        kl = float(np.mean(np.sum(pi * (np.log(pi + 1e-12) - np.log(pi_ref + 1e-12)), axis=1)))

        reward_history.append(expected_reward)
        kl_history.append(kl)

        logits = rlhf_policy_gradient_step(logits, r_phi, pi_ref, beta, lr)

    final_pi = softmax(logits, axis=-1)
    return final_pi, logits, reward_history, kl_history


def closed_form_optimal_policy(r_phi, pi_ref, beta):
    """
    KL-düzenlemeli amaç fonksiyonunun KAPALI FORM çözümü:

        pi*(k) ∝ pi_ref(k) * exp( r_phi(k) / beta )

    Bu formül, RLHF ile DPO arasındaki teorik köprüdür — DPO makalesi,
    bu kapalı formu tersine çevirerek ödülü politika cinsinden ifade eder
    ve reward-model + RL adımlarını tamamen ortadan kaldırır.
    Burada bu kapalı formu, gradyan çıkışı ile bulduğumuz sayısal çözümü
    doğrulamak için kullanıyoruz.
    """
    unnormalized = pi_ref * np.exp(r_phi / beta)
    return unnormalized / np.sum(unnormalized, axis=-1, keepdims=True)
