"""
test_core.py
------------
DPO ve RLHF gradyan türetimlerini SAYISAL GRADYAN KONTROLÜ (numerical
gradient checking) ile doğrular. Bu, elle türetilmiş gradyan formüllerinin
gerçekten doğru olduğunu kanıtlamanın standart yoludur: analitik gradyanı,
küçük bir epsilon ile hesaplanan (f(x+eps)-f(x-eps))/(2*eps) sayısal
gradyanıyla karşılaştırırız.

Çalıştırma: python test_core.py
"""

import numpy as np
from environment import build_environment, sigmoid
from reward_model import train_reward_model, reward_model_accuracy
from rlhf import softmax, train_rlhf_policy, closed_form_optimal_policy
from dpo import dpo_loss_and_grad, train_dpo_policy


def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + eps
        f_plus = f(x)

        x[idx] = orig - eps
        f_minus = f(x)

        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def test_dpo_gradient_matches_numerical():
    rng = np.random.default_rng(1)
    n_prompts, k_responses = 3, 4
    pi_ref = np.full((n_prompts, k_responses), 1.0 / k_responses)
    preferences = [(0, 1, 2), (0, 3, 0), (1, 0, 1), (2, 2, 3)]
    beta = 0.7

    logits = rng.normal(0, 0.3, size=(n_prompts, k_responses))

    _, analytic_grad = dpo_loss_and_grad(logits, pi_ref, preferences, beta)

    def loss_fn(l):
        loss, _ = dpo_loss_and_grad(l, pi_ref, preferences, beta)
        return loss

    numeric_grad = numerical_gradient(loss_fn, logits.copy())

    max_diff = np.max(np.abs(analytic_grad - numeric_grad))
    assert max_diff < 1e-4, f"DPO gradyanı sayısal kontrolden geçemedi: max_diff={max_diff}"


def test_reward_model_learns_true_ordering_direction():
    env = build_environment(seed=0, num_pairs_per_prompt=60)
    r_phi, loss_hist = train_reward_model(env.preferences, 5, 6, lr=0.5, epochs=200)

    # Kayıp azalmalı (öğrenme gerçekleşiyor mu?)
    assert loss_hist[-1] < loss_hist[0], "Ödül modeli kaybı azalmadı"

    # Ödül modeli tercih verisini makul oranda doğru tahmin etmeli
    acc = reward_model_accuracy(r_phi, env.preferences)
    # Not: Bradley-Terry tercih verisi doğası gereği gürültülüdür (insan
    # etiketleyicilerin tutarsızlığını simüler), bu yüzden %100 doğruluk
    # beklenmez/istenmez (aşırı öğrenme belirtisi olurdu). Makul bir eşik yeterli.
    assert acc > 0.68, f"Ödül modeli doğruluğu çok düşük: {acc}"


def test_rlhf_converges_to_closed_form_solution():
    rng = np.random.default_rng(2)
    n_prompts, k_responses = 4, 5
    r_phi = rng.normal(0, 1, size=(n_prompts, k_responses))
    pi_ref = np.full((n_prompts, k_responses), 1.0 / k_responses)
    beta = 0.6

    final_pi, _, _, _ = train_rlhf_policy(r_phi, pi_ref, beta=beta, lr=0.5, epochs=800)
    target_pi = closed_form_optimal_policy(r_phi, pi_ref, beta)

    max_diff = np.max(np.abs(final_pi - target_pi))
    assert max_diff < 0.01, f"RLHF politikası kapalı-form çözüme yakınsamadı: max_diff={max_diff}"


def test_dpo_policy_is_valid_distribution():
    env = build_environment(seed=3, num_pairs_per_prompt=40)
    final_pi, _, loss_hist = train_dpo_policy(env.pi_ref, env.preferences, beta=0.5, lr=0.3, epochs=200)

    assert np.allclose(np.sum(final_pi, axis=1), 1.0, atol=1e-6), "Politika satırları 1'e toplanmıyor"
    assert np.all(final_pi >= 0), "Negatif olasılık üretildi"
    assert loss_hist[-1] < loss_hist[0], "DPO kaybı azalmadı"


def _run_all():
    tests = [
        test_dpo_gradient_matches_numerical,
        test_reward_model_learns_true_ordering_direction,
        test_rlhf_converges_to_closed_form_solution,
        test_dpo_policy_is_valid_distribution,
    ]
    for t in tests:
        t()
        print(f"OK  - {t.__name__}")
    print("\nTüm testler başarıyla geçti.")


if __name__ == "__main__":
    _run_all()
