import pathlib

import pandas as pd


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TO_LABEL = DATA_DIR / "to_label.csv"
OUTPUT = DATA_DIR / "labeled.csv"


def main():
    if not TO_LABEL.exists():
        print("Файл data/to_label.csv не найден.")
        print("Сначала запустите 02_make_label_template.py.")
        return

    df = pd.read_csv(TO_LABEL)

    if "label" not in df.columns or "text" not in df.columns:
        print("В файле должны быть колонки text и label.")
        return

    df = df.dropna(subset=["text", "label"])

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]

    df = df.drop_duplicates(subset=["text"])

    if len(df) < 20:
        print("Слишком мало размеченных текстов.")
        print("Желательно иметь минимум 50 текстов каждого класса.")

    df.to_csv(OUTPUT, index=False)

    print("Размеченный датасет сохранен:", OUTPUT)
    print("Распределение меток:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
