"""
Titanic ML Dashboard - Streamlit Web Arayüzü
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "/app/models"
DATA_DIR  = "/app/data"

# ── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Stil ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card{background:#1e293b;border-radius:12px;padding:1.2rem 1.5rem;
               margin:.4rem 0;border-left:4px solid #3b82f6;}
  .metric-val {font-size:2rem;font-weight:700;color:#f1f5f9;}
  .metric-lbl {font-size:.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;}
  .winner-badge{background:#10b981;color:#fff;padding:.2rem .7rem;
                border-radius:99px;font-size:.75rem;font-weight:600;margin-left:.5rem;}
  h1{color:#f1f5f9 !important;}
</style>
""", unsafe_allow_html=True)


# ── Yardımcı fonksiyonlar ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)

    models, scaler = {}, None
    name_map = {"Random Forest": "random_forest", "XGBoost": "xgboost", "SVM": "svm"}

    for display, fname in name_map.items():
        path = os.path.join(MODEL_DIR, f"{fname}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[display] = pickle.load(f)

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return models, scaler, meta


def metric_card(label, value, suffix=""):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-lbl">{label}</div>
      <div class="metric-val">{value}{suffix}</div>
    </div>""", unsafe_allow_html=True)


# ── Ana arayüz ───────────────────────────────────────────────────────────────
st.title("🚢 Titanic Survival — ML Model Karşılaştırma Paneli")
st.caption("Random Forest · XGBoost · SVM | Titanic veri seti üzerinde eğitilmiş modeller")

models, scaler, meta = load_artifacts()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Navigasyon")
    page = st.radio("Sayfa", ["📊 Model Karşılaştırma", "🔍 Canlı Tahmin", "📈 Veri Analizi"])

    st.divider()
    if meta:
        st.metric("Eğitim Seti", f"{meta['train_size']} satır")
        st.metric("Test Seti",   f"{meta['test_size']} satır")

# ────────────────────────────────────────────────────────────────────────────
if page == "📊 Model Karşılaştırma":
    if not meta:
        st.warning("⚠️ Modeller henüz eğitilmedi. `python src/train.py` komutunu çalıştırın.")
        st.stop()

    metrics_df = pd.DataFrame(meta["metrics"])

    # En iyi model
    best_idx   = metrics_df["f1"].idxmax()
    best_model = metrics_df.loc[best_idx, "model"]

    st.subheader("🏆 Genel Başarı Metrikleri")
    cols = st.columns(len(metrics_df))
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        badge = '<span class="winner-badge">BEST</span>' if row["model"] == best_model else ""
        with cols[i]:
            st.markdown(f"### {row['model']}{badge}", unsafe_allow_html=True)
            metric_card("Accuracy",  f"{row['accuracy']*100:.1f}", "%")
            metric_card("F1-Score",  f"{row['f1']*100:.1f}",      "%")
            metric_card("ROC-AUC",   f"{row['roc_auc']*100:.1f}", "%")
            metric_card("Precision", f"{row['precision']*100:.1f}","%")
            metric_card("Recall",    f"{row['recall']*100:.1f}",   "%")
            if "cv_mean" in row and pd.notna(row["cv_mean"]):
                metric_card("CV (5-fold)", f"{row['cv_mean']*100:.1f}", "%")

    st.divider()

    # Bar grafik karşılaştırması
    st.subheader("📊 Metrik Karşılaştırması")
    plot_cols = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    melted = metrics_df.melt(id_vars="model", value_vars=plot_cols,
                              var_name="Metrik", value_name="Değer")
    melted["Metrik"] = melted["Metrik"].str.upper().str.replace("_", "-")

    fig = px.bar(
        melted, x="Metrik", y="Değer", color="model", barmode="group",
        color_discrete_sequence=["#3b82f6","#10b981","#f59e0b"],
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        legend_title_text="Model", yaxis_tickformat=".0%",
        yaxis_range=[0.5, 1.0], height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    st.subheader("🔢 Confusion Matrix")
    cm_cols = st.columns(len(metrics_df))
    for i, row in metrics_df.iterrows():
        cm = np.array(row["confusion_matrix"])
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Tahmin", y="Gerçek", color="Sayı"),
            x=["Hayatta Kalmadı","Hayatta Kaldı"],
            y=["Hayatta Kalmadı","Hayatta Kaldı"],
            color_continuous_scale="Blues", template="plotly_dark",
            title=row["model"],
        )
        fig_cm.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            height=280, margin=dict(t=40, b=20, l=20, r=20),
            coloraxis_showscale=False,
        )
        cm_cols[i].plotly_chart(fig_cm, use_container_width=True)

    # CV Scores
    cv_rows = metrics_df.dropna(subset=["cv_mean"])
    if not cv_rows.empty:
        st.subheader("📉 Cross-Validation (5-Fold)")
        fig_cv = go.Figure()
        for _, row in cv_rows.iterrows():
            fig_cv.add_trace(go.Bar(
                name=row["model"], x=[row["model"]],
                y=[row["cv_mean"]], error_y=dict(type="data", array=[row["cv_std"]]),
            ))
        fig_cv.update_layout(
            template="plotly_dark", plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            yaxis_tickformat=".0%", yaxis_range=[0.5, 1.0], height=350,
            showlegend=False,
        )
        st.plotly_chart(fig_cv, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Canlı Tahmin":
    if not models:
        st.warning("⚠️ Modeller yüklenemedi. Önce `train.py` çalıştırın.")
        st.stop()

    st.subheader("🧑 Yolcu Bilgilerini Girin")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            pclass   = st.selectbox("Bilet Sınıfı (Pclass)", [1, 2, 3], index=2)
            sex      = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
            age      = st.slider("Yaş", 1, 80, 28)
        with c2:
            fare     = st.slider("Bilet Ücreti ($)", 0, 520, 32)
            embarked = st.selectbox("Biniş Limanı", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
            sibsp    = st.number_input("Kardeş/Eş Sayısı (SibSp)", 0, 8, 0)
        with c3:
            parch      = st.number_input("Ebeveyn/Çocuk Sayısı (Parch)", 0, 6, 0)
            title_opt  = st.selectbox("Unvan", ["Mr", "Miss", "Mrs", "Master", "Rare"])
            chosen_mdl = st.selectbox("Model", list(models.keys()))

        submitted = st.form_submit_button("🔮 Tahmin Et", use_container_width=True)

    if submitted:
        sex_val      = 1 if sex == "Erkek" else 0
        embarked_val = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}[embarked]
        title_val    = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}[title_opt]
        family_size  = sibsp + parch + 1
        is_alone     = 1 if family_size == 1 else 0

        features = np.array([[pclass, sex_val, age, fare, embarked_val,
                               family_size, is_alone, title_val, sibsp, parch]])

        model = models[chosen_mdl]
        if chosen_mdl == "SVM" and scaler:
            features_input = scaler.transform(features)
        else:
            features_input = features

        prediction = model.predict(features_input)[0]
        probability = model.predict_proba(features_input)[0] if hasattr(model, "predict_proba") else None

        st.divider()
        if prediction == 1:
            st.success("✅ **Hayatta Kalma İhtimali: YÜKSEK**")
        else:
            st.error("❌ **Hayatta Kalma İhtimali: DÜŞÜK**")

        if probability is not None:
            p_survive = probability[1] * 100
            p_die     = probability[0] * 100
            col_a, col_b = st.columns(2)
            col_a.metric("Hayatta Kalma Olasılığı", f"{p_survive:.1f}%")
            col_b.metric("Hayatta Kalamama Olasılığı", f"{p_die:.1f}%")

            fig_p = go.Figure(go.Bar(
                x=["Hayatta Kalamaz","Hayatta Kalır"],
                y=[p_die, p_survive],
                marker_color=["#ef4444","#10b981"],
                text=[f"{p_die:.1f}%", f"{p_survive:.1f}%"],
                textposition="outside",
            ))
            fig_p.update_layout(
                template="plotly_dark", plot_bgcolor="#0f172a",
                paper_bgcolor="#0f172a", height=300,
                yaxis_range=[0, 110], showlegend=False,
            )
            st.plotly_chart(fig_p, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
elif page == "📈 Veri Analizi":
    csv_path = os.path.join(DATA_DIR, "titanic.csv")
    if not os.path.exists(csv_path):
        st.warning("⚠️ Veri seti bulunamadı. Önce modelleri eğitin.")
        st.stop()

    df = pd.read_csv(csv_path)

    st.subheader("📋 Veri Seti Önizleme")
    st.dataframe(df.head(20), use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Hayatta Kalma Dağılımı")
        vc = df["Survived"].value_counts().reset_index()
        vc.columns = ["Durum","Sayı"]
        vc["Durum"] = vc["Durum"].map({0:"Hayatta Kalmadı", 1:"Hayatta Kaldı"})
        fig1 = px.pie(vc, values="Sayı", names="Durum",
                      color_discrete_sequence=["#ef4444","#10b981"],
                      template="plotly_dark", hole=0.4)
        fig1.update_layout(paper_bgcolor="#0f172a")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Sınıf & Cinsiyet Bazlı Hayatta Kalma")
        grp = df.groupby(["Pclass","Sex"])["Survived"].mean().reset_index()
        fig2 = px.bar(grp, x="Pclass", y="Survived", color="Sex", barmode="group",
                      color_discrete_sequence=["#3b82f6","#f59e0b"],
                      template="plotly_dark", labels={"Survived":"Oran","Pclass":"Bilet Sınıfı"})
        fig2.update_layout(paper_bgcolor="#0f172a", yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Yaş Dağılımı (Survived vs Not)")
    fig3 = px.histogram(df.dropna(subset=["Age"]), x="Age", color="Survived",
                        nbins=40, barmode="overlay",
                        color_discrete_map={0:"#ef4444", 1:"#10b981"},
                        template="plotly_dark",
                        labels={"Survived":"Durum","Age":"Yaş"})
    fig3.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    st.plotly_chart(fig3, use_container_width=True)

    # Korelasyon ısı haritası
    st.subheader("Korelasyon Matrisi")
    num_df = df[["Survived","Pclass","Age","SibSp","Parch","Fare"]].dropna()
    fig4 = px.imshow(num_df.corr(), text_auto=".2f",
                     color_continuous_scale="RdBu_r", template="plotly_dark")
    fig4.update_layout(paper_bgcolor="#0f172a", height=420)
    st.plotly_chart(fig4, use_container_width=True)
