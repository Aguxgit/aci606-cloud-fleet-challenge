"""Ejercicio 4: entrenamiento reproducible del clasificador de condición."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["speed", "battery", "motor_temp", "vibration",
            "motor_current", "obstacle_distance"]
TARGET = "status"
LABELS = ["NORMAL", "WARNING", "CRITICAL"]
VERSION = "v2"


def load_data() -> tuple[pd.DataFrame, str]:
    files = sorted(Path("data").glob("training*.csv"))
    if not files:
        raise SystemExit("Falta el dataset: python simulator/generate_dataset.py")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if "fleet_id" not in d.columns:
            d["fleet_id"] = f.stem          # cada archivo = una flota
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if df["fleet_id"].nunique() >= 2:
        return df, "fleet_id"
    print("AVISO: una sola flota; se separa por robot_id (sustituto de fleet_id).")
    return df, "robot_id"


def main() -> None:
    df, group_col = load_data()
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"Archivos/flotas: {df['fleet_id'].nunique()} | grupos para el split: {group_col}")
    print("\nDistribución de clases:")
    print(df[TARGET].value_counts())
    print(df[TARGET].value_counts(normalize=True).round(3))

    X, y, g = df[FEATURES], df[TARGET], df[group_col]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                    random_state=42).split(X, y, g))
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    print(f"\nTrain: {len(X_tr)} filas | Test: {len(X_te)} filas (grupos no vistos)")

    models = {
        "baseline_mayoritaria": DummyClassifier(strategy="most_frequent"),
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000,
                                                       class_weight="balanced"))]),
        "random_forest": Pipeline([("clf", RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1))]),
    }

    scores = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        scores[name] = round(f1_score(y_te, pred, average="macro",
                                      labels=LABELS, zero_division=0), 4)
        print(f"\n=== {name} | F1 macro = {scores[name]}")
        print(confusion_matrix(y_te, pred, labels=LABELS))
        print(classification_report(y_te, pred, labels=LABELS, zero_division=0))

    best = max((k for k in scores if k != "baseline_mayoritaria"), key=scores.get)
    print(f"\nModelo elegido: {best} (F1 macro {scores[best]})")

    final = models[best]
    final.fit(X, y)  # reentrenar con todos los datos antes de guardar
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": final, "features": FEATURES, "version": VERSION},
                "models/model.joblib")
    Path("models/metrics.json").write_text(json.dumps(
        {"elegido": best, "f1_macro": scores, "split": group_col}, indent=2))
    print("Guardado en models/model.joblib")


if __name__ == "__main__":
    main()