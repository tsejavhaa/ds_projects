# collector_snscrape.py
"""
Scrape tweets using snscrape and classify/store them.
No API key required.
"""

import snscrape.modules.twitter as sntwitter
import sqlite3
from transformers import pipeline
from datetime import datetime
import logging

DB_PATH = "tweets.db"
TABLE_NAME = "tweets"
KEYWORD = "tesla"  # change to your product or hashtag
MAX_TWEETS = 5000

logging.basicConfig(level=logging.INFO)
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            username TEXT,
            text TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            collected_at TEXT
        )
        """
    )
    conn.commit()
    return conn

def insert_tweet(conn, record):
    placeholders = ", ".join("?" * len(record))
    columns = ", ".join(record.keys())
    sql = f"INSERT OR IGNORE INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})"
    cur = conn.cursor()
    cur.execute(sql, tuple(record.values()))
    conn.commit()

def main():
    conn = init_db()
    query = f"{KEYWORD} lang:en"
    logging.info("Starting snscrape for query: %s", query)
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= MAX_TWEETS:
            break
        text = tweet.content
        res = sentiment_pipe(text[:512])
        record = {
            "id": str(tweet.id),
            "created_at": str(tweet.date),
            "username": tweet.user.username,
            "text": text,
            "sentiment_label": res[0]["label"],
            "sentiment_score": float(res[0]["score"]),
            "collected_at": datetime.utcnow().isoformat(),
        }
        insert_tweet(conn, record)
        if i % 50 == 0:
            logging.info("Scraped %d tweets", i)

if __name__ == "__main__":
    main()