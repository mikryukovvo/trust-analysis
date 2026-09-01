import pathlib
import hashlib
import datetime

import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

FEEDS_FILE = BASE_DIR / "feeds.txt"
OUTPUT_FILE = DATA_DIR / "raw_rss.csv"

KEYWORDS = [
    "власть",
    "чиновник",
    "администрация",
    "правительство",
    "госуслуги",
    "государственные услуги",
    "жалоба",
    "обращение",
    "ремонт",
    "жкх",
    "отопление",
    "школа",
    "больница",
    "поликлиника",
    "врач",
    "дороги",
    "транспорт",
    "социальные выплаты",
    "субсидии",
    "тарифы",
    "регион",
    "губернатор",
    "мэрия",
    "муниципалитет",
    "доверие",
    "недоверие",
]


def clean_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text.strip()


def load_feeds() -> list[str]:
    if not FEEDS_FILE.exists():
        return []

    feeds = []
    with open(FEEDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                feeds.append(line)
    return feeds


def collect():
    feeds = load_feeds()

    if not feeds:
        print("Файл feeds.txt пуст. Добавьте RSS-ссылки.")
        return

    records = []

    headers = {
        "User-Agent": "Research pilot RSS collector; contact: your-email@example.com"
    }

    for feed_url in feeds:
        print(f"Обработка: {feed_url}")

        try:
            response = requests.get(feed_url, timeout=20, headers=headers)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)

            for entry in parsed.entries:
                title = clean_html(entry.get("title", ""))
                summary = clean_html(entry.get("summary", ""))

                text = f"{title}. {summary}".strip()
                if len(text) < 20:
                    continue

                text_lower = text.lower()

                if any(keyword in text_lower for keyword in KEYWORDS):
                    text_id = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

                    records.append(
                        {
                            "id": text_id,
                            "text": text[:500],
                            "source_feed": feed_url,
                            "published": entry.get("published", ""),
                            "collected_at": datetime.datetime.utcnow().isoformat(),
                        }
                    )

        except Exception as exc:
            print(f"Ошибка при обработке {feed_url}: {exc}")

    df = pd.DataFrame(records)

    if df.empty:
        print("Ничего не собрано. Проверьте ссылки в feeds.txt.")
        df.to_csv(OUTPUT_FILE, index=False)
        return

    df = df.drop_duplicates(subset=["text"])
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Собрано строк: {len(df)}")
    print(f"Файл сохранен: {OUTPUT_FILE}")


if __name__ == "__main__":
    collect()
