"""
environment.py
---------------
RLHF ve DPO'yu aynı sentetik ortam üzerinde karşılaştırmak için kullanılan
basit bir "bandit" tarzı ortam.

Kurgu:
  - N_PROMPTS farklı "prompt" (bağlam) var.
  - Her prompt için K_RESPONSES farklı olası "cevap" var (örn. bir dil modelinin
    üretebileceği K farklı tamamlama).
  - Her (prompt, cevap) çiftinin gizli/bilinmeyen bir GERÇEK kalite skoru
    (true_quality) vardır — gerçek hayatta bu, bir insan değerlendiricinin
    zihnindeki "ne kadar iyi bir cevap" yargısına karşılık gelir ve modele
    asla doğrudan verilmez.
  - Bu gerçek kaliteden, Bradley-Terry modeliyle GÜRÜLTÜLÜ ikili tercih
    verisi (preference data) üretilir: "cevap i, cevap j'ye tercih edildi"
    şeklinde insan geri bildirimini simüle eder.

Bradley-Terry modeli:
    P(i, j'ye tercih edilir) = sigmoid(quality_i - quality_j)

Bu tam olarak RLHF/DPO makalelerinde insan tercihlerini modellemek için
kullanılan istatistiksel modeldir.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


N_PROMPTS = 5
K_RESPONSES = 6


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Environment:
    true_quality: np.ndarray                 # (N_PROMPTS, K_RESPONSES)
    preferences: List[Tuple[int, int, int]]   # (prompt, chosen, rejected)
    pi_ref: np.ndarray = field(init=False)    # referans (SFT) politikası

    def __post_init__(self):
        # Referans politika: hizalama öncesi model, her cevaba eşit olasılık verir
        # (gerçek dünyada bu, SFT modelinin ürettiği dağılımdır).
        self.pi_ref = np.full((N_PROMPTS, K_RESPONSES), 1.0 / K_RESPONSES)


def build_environment(seed: int = 0, num_pairs_per_prompt: int = 40) -> Environment:
    rng = np.random.default_rng(seed)

    # Her (prompt, cevap) için gizli kalite skoru. Biraz prompt-bazlı ölçek
    # farkı da ekleyelim ki ortam gerçekçi (heterojen) olsun.
    true_quality = rng.normal(loc=0.0, scale=1.0, size=(N_PROMPTS, K_RESPONSES))

    preferences = []
    for p in range(N_PROMPTS):
        for _ in range(num_pairs_per_prompt):
            i, j = rng.choice(K_RESPONSES, size=2, replace=False)
            p_i_wins = sigmoid(true_quality[p, i] - true_quality[p, j])
            if rng.random() < p_i_wins:
                preferences.append((p, int(i), int(j)))   # i tercih edildi
            else:
                preferences.append((p, int(j), int(i)))   # j tercih edildi

    return Environment(true_quality=true_quality, preferences=preferences)


def expected_true_quality(pi: np.ndarray, true_quality: np.ndarray) -> float:
    """Bir politikanın, gerçek (gizli) kaliteye göre beklenen skorunu hesaplar.
    Bu, gerçek hayatta asla doğrudan optimize edilemeyen ama bizim
    'gerçekten ne kadar iyi hizalandık?' diye ölçmek için kullandığımız
    üst-sınır (oracle) metriktir."""
    return float(np.mean(np.sum(pi * true_quality, axis=1)))


def kl_divergence(pi: np.ndarray, pi_ref: np.ndarray) -> float:
    """KL(pi || pi_ref), prompt başına ortalanmış."""
    eps = 1e-12
    kl_per_prompt = np.sum(pi * (np.log(pi + eps) - np.log(pi_ref + eps)), axis=1)
    return float(np.mean(kl_per_prompt))


def pairwise_accuracy(pi: np.ndarray, true_quality: np.ndarray) -> float:
    """Politikanın en yüksek olasılık verdiği cevabın, gerçekte de en kaliteli
    cevap olma oranı (prompt'lar üzerinden). Basit ama anlaşılır bir metrik."""
    pred_best = np.argmax(pi, axis=1)
    true_best = np.argmax(true_quality, axis=1)
    return float(np.mean(pred_best == true_best))
