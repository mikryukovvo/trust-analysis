import pathlib

import numpy as np
import pandas as pd
import shap

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LABELED_CSV = DATA_DIR / "labeled.csv"


def main():
    if not LABELED_CSV.exists():
        print("Файл data/labeled.csv не найден.")
        return

    df = pd.read_csv(LABELED_CSV)

    if len(df) < 20:
        print("Слишком мало данных для SHAP-анализа.")
        return

    texts = df["text"].astype(str).fillna("")
    y = df["label"].astype(int).values

    if len(np.unique(y)) < 2:
        print("Нужны оба класса: 0 и 1.")
        return

    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X, y)

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[2] > 1:
            values = shap_values[:, :, 1]
        else:
            values = shap_values[:, :, 0]
    elif shap_values.ndim == 2:
        values = shap_values
    else:
        values = shap_values.reshape(-1, 1)

    mean_abs_shap = np.abs(values).mean(axis=0)

    features = vectorizer.get_feature_names_out()

    top_df = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": mean_abs_shap
        }
    )

    top_df = top_df.sort_values("mean_abs_shap", ascending=False)
    top_df.to_csv(RESULTS_DIR / "shap_top_features.csv", index=False)

    print("SHAP-анализ завершен.")
    print("Топ признаков сохранен в results/shap_top_features.csv")
    print(top_df.head(20))


if __name__ == "__main__":
    main()
