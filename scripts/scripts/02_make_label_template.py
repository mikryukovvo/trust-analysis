import pathlib

import pandas as pd


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_RSS = DATA_DIR / "raw_rss.csv"
MANUAL = DATA_DIR / "manual_texts.csv"
OUTPUT = DATA_DIR / "to_label.csv"


def main():
    if RAW_RSS.exists():
        df = pd.read_csv(RAW_RSS)
        if "text" in df.columns and len(df) > 0:
            df = df[["id", "text"]].copy()
            df = df.drop_duplicates(subset=["text"])
            df = df.head(200)
        else:
            df = pd.DataFrame(columns=["id", "text"])
    else:
        df = pd.DataFrame(columns=["id", "text"])

    if len(df) == 0 and MANUAL.exists():
        manual_df = pd.read_csv(MANUAL)
        if "text" in manual_df.columns:
            if "id" not in manual_df.columns:
                manual_df["id"] = range(1, len(manual_df) + 1)
            df = manual_df[["id", "text"]].copy()

    if len(df) == 0:
        print("Нет данных для разметки.")
        print("Соберите RSS через 01_collect_rss.py или создайте data/manual_texts.csv.")
        return

    df["label"] = ""
    df["notes"] = ""

    df.to_csv(OUTPUT, index=False)

    print(f"Создан файл для разметки: {OUTPUT}")
    print("Заполните колонку label:")
    print("1 = доверие / поддержка / положительная оценка")
    print("0 = недоверие / критика / жалоба")
    print("Нейтральные тексты оставляйте пустыми.")


if __name__ == "__main__":
    main()
