"""
demo.py
-------
Projenin ana çalıştırılabilir dosyası.

Aynı sentetik tercih veri kümesi üzerinde İKİ farklı hizalama (alignment)
yaklaşımını uçtan uca çalıştırır ve karşılaştırır:

  YOL A) Klasik RLHF:  tercih verisi -> ödül modeli eğit -> KL-düzenlemeli
                        politika optimizasyonu (PPO'nun tam-gradyanlı hali)
  YOL B) DPO:           tercih verisi -> politikayı DOĞRUDAN optimize et
                        (ödül modeli YOK, ayrı RL adımı YOK)

Ardından iki yolun:
  - gerçek (gizli) kaliteye göre beklenen skorunu,
  - referans politikadan KL uzaklığını,
  - en iyi cevabı doğru tahmin etme oranını (pairwise accuracy),
  - ve DPO'nun örtük ödülünün, RLHF'in öğrendiği ödül modeliyle
    ne kadar örtüştüğünü

karşılaştırır. Bu son karşılaştırma, DPO'nun teorik iddiasını doğrular:
"DPO, ayrı bir ödül modeli eğitmeden, RLHF'in bulacağı ödülle
UYUMLU bir politikaya ulaşır."

Çalıştırma:
    python demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import build_environment, expected_true_quality, kl_divergence, pairwise_accuracy
from reward_model import train_reward_model, reward_model_accuracy
from rlhf import train_rlhf_policy, closed_form_optimal_policy, softmax
from dpo import train_dpo_policy, implicit_reward

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BETA = 0.5           # KL/DPO düzenleme katsayısı (iki yöntemde de AYNI beta kullanılır!)
EPOCHS = 400
LR = 0.4


def separator(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    # ---------------------------------------------------------------
    # 0) Ortamı ve gürültülü insan tercih verisini üret
    # ---------------------------------------------------------------
    separator("0) SENTETİK ORTAM VE İNSAN TERCİH VERİSİ")
    env = build_environment(seed=0, num_pairs_per_prompt=50)
    n_prompts, k_responses = env.true_quality.shape
    print(f"Prompt sayısı            : {n_prompts}")
    print(f"Prompt başına cevap sayısı: {k_responses}")
    print(f"Toplam tercih çifti       : {len(env.preferences)}")
    print(f"Referans politika (pi_ref): uniform (SFT sonrası, hizalanmamış model)")

    # ---------------------------------------------------------------
    # YOL A: KLASİK RLHF
    # ---------------------------------------------------------------
    separator("YOL A) RLHF — Adım 1: Ödül Modeli Eğitimi (Bradley-Terry)")
    r_phi, rm_loss_hist = train_reward_model(
        env.preferences, n_prompts, k_responses, lr=0.7, epochs=300
    )
    rm_acc = reward_model_accuracy(r_phi, env.preferences)
    print(f"Ödül modeli eğitim kaybı : {rm_loss_hist[0]:.4f} -> {rm_loss_hist[-1]:.4f}")
    print(f"Ödül modeli doğruluğu    : {rm_acc * 100:.1f}% (tercih verisi üzerinde)")

    separator("YOL A) RLHF — Adım 2: KL-Düzenlemeli Politika Optimizasyonu")
    rlhf_pi, rlhf_logits, rlhf_reward_hist, rlhf_kl_hist = train_rlhf_policy(
        r_phi, env.pi_ref, beta=BETA, lr=LR, epochs=EPOCHS
    )
    closed_form_pi = closed_form_optimal_policy(r_phi, env.pi_ref, BETA)
    closed_form_diff = float(np.max(np.abs(rlhf_pi - closed_form_pi)))
    print(f"Ödül modeline göre beklenen ödül : {rlhf_reward_hist[-1]:.4f}")
    print(f"Referanstan KL uzaklığı          : {rlhf_kl_hist[-1]:.4f}")
    print(f"Kapalı-form çözümden max sapma   : {closed_form_diff:.6f}  "
          f"(çok küçükse, gradyan çıkışı teorik optimuma ulaşmış demektir)")

    # ---------------------------------------------------------------
    # YOL B: DPO
    # ---------------------------------------------------------------
    separator("YOL B) DPO — Tek Adımda Doğrudan Politika Optimizasyonu")
    dpo_pi, dpo_logits, dpo_loss_hist = train_dpo_policy(
        env.pi_ref, env.preferences, beta=BETA, lr=LR, epochs=EPOCHS
    )
    print(f"DPO eğitim kaybı: {dpo_loss_hist[0]:.4f} -> {dpo_loss_hist[-1]:.4f}")
    print("(Not: Ödül modeli eğitimi YOK, ayrı RL adımı YOK — tek bir kayıp fonksiyonu.)")

    # ---------------------------------------------------------------
    # KARŞILAŞTIRMA
    # ---------------------------------------------------------------
    separator("KARŞILAŞTIRMA: RLHF vs. DPO vs. Referans Politika")

    header = f"{'Metrik':<38}{'Referans':>12}{'RLHF':>12}{'DPO':>12}"
    print(header)
    print("-" * len(header))

    def row(name, ref_val, rlhf_val, dpo_val, fmt="{:.4f}"):
        print(f"{name:<38}{fmt.format(ref_val):>12}{fmt.format(rlhf_val):>12}{fmt.format(dpo_val):>12}")

    eq_ref = expected_true_quality(env.pi_ref, env.true_quality)
    eq_rlhf = expected_true_quality(rlhf_pi, env.true_quality)
    eq_dpo = expected_true_quality(dpo_pi, env.true_quality)
    row("Beklenen GERÇEK kalite (oracle)", eq_ref, eq_rlhf, eq_dpo)

    kl_ref = 0.0
    kl_rlhf = kl_divergence(rlhf_pi, env.pi_ref)
    kl_dpo = kl_divergence(dpo_pi, env.pi_ref)
    row("Referanstan KL uzaklığı", kl_ref, kl_rlhf, kl_dpo)

    acc_ref = pairwise_accuracy(env.pi_ref, env.true_quality)
    acc_rlhf = pairwise_accuracy(rlhf_pi, env.true_quality)
    acc_dpo = pairwise_accuracy(dpo_pi, env.true_quality)
    row("En iyi cevabı doğru bulma oranı", acc_ref, acc_rlhf, acc_dpo)

    # DPO'nun örtük ödülü ile RLHF'in öğrendiği ödül modeli ne kadar örtüşüyor?
    dpo_implicit_r = implicit_reward(dpo_logits, env.pi_ref, BETA)
    # İki ödül fonksiyonu farklı bir sabitle kayabilir (Bradley-Terry kaymaya
    # duyarsızdır), bu yüzden korelasyona bakıyoruz, mutlak değere değil.
    correlation = float(np.corrcoef(r_phi.flatten(), dpo_implicit_r.flatten())[0, 1])
    print(f"\nRLHF ödül modeli <-> DPO örtük ödülü korelasyonu: {correlation:.4f}  "
          f"(1.0'a yakınsa, iki yöntem aynı ödül yapısını öğrenmiş demektir)")

    pi_diff = float(np.mean(np.abs(rlhf_pi - dpo_pi)))
    print(f"RLHF politikası <-> DPO politikası ort. mutlak fark: {pi_diff:.4f}")

    plot_results(env, rlhf_reward_hist, rlhf_kl_hist, rm_loss_hist, dpo_loss_hist,
                 rlhf_pi, dpo_pi, r_phi, dpo_implicit_r)
    write_report(env, rm_acc, rlhf_pi, dpo_pi, eq_ref, eq_rlhf, eq_dpo,
                 kl_rlhf, kl_dpo, acc_ref, acc_rlhf, acc_dpo, correlation, pi_diff)

    separator("TAMAMLANDI")
    print(f"Tüm çıktılar '{os.path.abspath(OUTPUT_DIR)}' klasöründe.")


def plot_results(env, rlhf_reward_hist, rlhf_kl_hist, rm_loss_hist, dpo_loss_hist,
                  rlhf_pi, dpo_pi, r_phi, dpo_implicit_r):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # (1) Ödül modeli eğitim kaybı
    axes[0, 0].plot(rm_loss_hist, color="#4C72B0")
    axes[0, 0].set_title("RLHF Adım 1: Ödül Modeli Kaybı")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Bradley-Terry Kaybı")
    axes[0, 0].grid(alpha=0.3)

    # (2) RLHF: beklenen ödül eğrisi
    axes[0, 1].plot(rlhf_reward_hist, color="#55A868", label="Beklenen ödül (r_phi)")
    axes[0, 1].set_title("RLHF Adım 2: Politika Optimizasyonu")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Beklenen ödül", color="#55A868")
    ax_kl = axes[0, 1].twinx()
    ax_kl.plot(rlhf_kl_hist, color="#C44E52", label="KL(pi||pi_ref)")
    ax_kl.set_ylabel("KL uzaklığı", color="#C44E52")
    axes[0, 1].grid(alpha=0.3)

    # (3) DPO kayıp eğrisi
    axes[0, 2].plot(dpo_loss_hist, color="#8172B2")
    axes[0, 2].set_title("DPO: Doğrudan Optimizasyon Kaybı\n(ödül modeli YOK)")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("DPO Kaybı")
    axes[0, 2].grid(alpha=0.3)

    # (4) Örnek bir prompt için politika karşılaştırması (bar chart)
    example_prompt = 0
    x = np.arange(env.true_quality.shape[1])
    width = 0.25
    axes[1, 0].bar(x - width, env.pi_ref[example_prompt], width, label="Referans (SFT)", color="#999999")
    axes[1, 0].bar(x, rlhf_pi[example_prompt], width, label="RLHF", color="#55A868")
    axes[1, 0].bar(x + width, dpo_pi[example_prompt], width, label="DPO", color="#8172B2")
    axes[1, 0].set_title(f"Prompt #{example_prompt}: Politika Karşılaştırması")
    axes[1, 0].set_xlabel("Cevap indeksi")
    axes[1, 0].set_ylabel("Olasılık")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # (5) Gerçek kalite vs öğrenilen ödüller (scatter)
    axes[1, 1].scatter(env.true_quality.flatten(), r_phi.flatten(),
                        color="#55A868", alpha=0.7, label="RLHF ödül modeli (r_phi)")
    axes[1, 1].scatter(env.true_quality.flatten(), dpo_implicit_r.flatten(),
                        color="#8172B2", alpha=0.7, marker="^", label="DPO örtük ödülü")
    axes[1, 1].set_title("Gerçek Kalite vs. Öğrenilen Ödüller")
    axes[1, 1].set_xlabel("Gerçek (gizli) kalite")
    axes[1, 1].set_ylabel("Öğrenilen ödül")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    # (6) RLHF ödülü vs DPO örtük ödülü (doğrudan karşılaştırma)
    axes[1, 2].scatter(r_phi.flatten(), dpo_implicit_r.flatten(), color="#4C72B0", alpha=0.7)
    lims = [min(r_phi.min(), dpo_implicit_r.min()), max(r_phi.max(), dpo_implicit_r.max())]
    axes[1, 2].plot(lims, lims, "--", color="gray", linewidth=1)
    axes[1, 2].set_title("RLHF Ödül Modeli vs. DPO Örtük Ödülü")
    axes[1, 2].set_xlabel("RLHF: r_phi(p,k)")
    axes[1, 2].set_ylabel("DPO: beta * log(pi/pi_ref)")
    axes[1, 2].grid(alpha=0.3)

    fig.suptitle("RLHF vs. DPO — Aynı Tercih Verisi Üzerinde Karşılaştırma", fontsize=15)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "rlhf_vs_dpo.png")
    fig.savefig(out_path, dpi=140)
    print(f"\n[Grafik kaydedildi] {out_path}")


def write_report(env, rm_acc, rlhf_pi, dpo_pi, eq_ref, eq_rlhf, eq_dpo,
                  kl_rlhf, kl_dpo, acc_ref, acc_rlhf, acc_dpo, correlation, pi_diff):
    lines = []
    lines.append("RLHF vs DPO DEMO RAPORU")
    lines.append("=" * 40)
    lines.append(f"Ödül modeli doğruluğu: {rm_acc*100:.1f}%")
    lines.append("")
    lines.append(f"{'Metrik':<38}{'Referans':>12}{'RLHF':>12}{'DPO':>12}")
    lines.append(f"{'Beklenen gerçek kalite':<38}{eq_ref:>12.4f}{eq_rlhf:>12.4f}{eq_dpo:>12.4f}")
    lines.append(f"{'KL uzaklığı':<38}{0.0:>12.4f}{kl_rlhf:>12.4f}{kl_dpo:>12.4f}")
    lines.append(f"{'En iyi cevabı bulma oranı':<38}{acc_ref:>12.4f}{acc_rlhf:>12.4f}{acc_dpo:>12.4f}")
    lines.append("")
    lines.append(f"RLHF ödül modeli <-> DPO örtük ödülü korelasyonu: {correlation:.4f}")
    lines.append(f"RLHF politikası <-> DPO politikası ort. mutlak fark: {pi_diff:.4f}")

    out_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Rapor kaydedildi] {out_path}")


if __name__ == "__main__":
    main()
