"""
Titanic ML Pipeline - Model Training
Trains Random Forest, XGBoost, and SVM on Titanic dataset
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
MODEL_DIR = "/app/models"
DATA_DIR  = "/app/data"


def load_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "titanic.csv")
    if not os.path.exists(csv_path):
        print("Veri seti indiriliyor...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
    print(f"Veri seti yüklendi: {df.shape}")
    return df


def preprocess(df):
    df = df.copy()

    # Feature engineering
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    df["Sex"]      = (df["Sex"] == "male").astype(int)
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    df["Title"]      = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"]      = df["Title"].replace(
        ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"], "Rare"
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}).fillna(0)

    features = ["Pclass","Sex","Age","Fare","Embarked",
                "FamilySize","IsAlone","Title","SibSp","Parch"]
    X = df[features]
    y = df["Survived"]
    return X, y, features


def evaluate(name, model, X_test, y_test, X_train=None, y_train=None):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    if X_train is not None and y_train is not None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        metrics["cv_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_std"]  = round(cv_scores.std(),  4)

    return metrics


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    X, y, features = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", verbosity=0
        ),
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=42
        ),
    }

    all_metrics = []
    for name, model in models.items():
        print(f"\n--- {name} eğitiliyor ---")
        if name == "SVM":
            model.fit(X_train_scaled, y_train)
            metrics = evaluate(name, model, X_test_scaled, y_test, X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
            metrics = evaluate(name, model, X_test, y_test, X_train, y_train)

        print(f"  Accuracy : {metrics['accuracy']}")
        print(f"  F1-Score : {metrics['f1']}")
        print(f"  ROC-AUC  : {metrics['roc_auc']}")

        # Save model
        safe_name = name.lower().replace(" ", "_")
        with open(os.path.join(MODEL_DIR, f"{safe_name}.pkl"), "wb") as f:
            pickle.dump(model, f)

        all_metrics.append(metrics)

    # Save scaler & metadata
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "features": features,
        "train_size": len(X_train),
        "test_size":  len(X_test),
        "metrics":    all_metrics,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Tüm modeller kaydedildi.")
    return all_metrics


if __name__ == "__main__":
    train()
