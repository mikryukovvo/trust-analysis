import json
import pathlib

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LABELED_CSV = DATA_DIR / "labeled.csv"


def main():
    if not LABELED_CSV.exists():
        print("Файл data/labeled.csv не найден.")
        print("Сначала выполните разметку и запустите 03_prepare_labeled.py.")
        return

    df = pd.read_csv(LABELED_CSV)

    if len(df) < 20:
        print("Слишком мало данных для пилота.")
        return

    texts = df["text"].astype(str).fillna("")
    y = df["label"].astype(int).values

    if len(np.unique(y)) < 2:
        print("Нужны оба класса: 0 и 1.")
        return

    class_counts = np.bincount(y)
    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        print("В каждом классе должно быть минимум 2 примера.")
        return

    n_splits = min(5, min_class_count)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    y_prob = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "n_samples": int(len(df)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "cv_folds": int(n_splits),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_df = df.copy()
    pred_df["predicted_label"] = y_pred
    pred_df["prob_positive"] = y_prob
    pred_df.to_csv(RESULTS_DIR / "cv_predictions.csv", index=False)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"]
    )
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")

    print("Метрики пилота:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Результаты сохранены в results/")


if __name__ == "__main__":
    main()
