# collector.py
"""
Twitter streaming collector + simple transformer sentiment classifier.
Saves results into a local SQLite DB (tweets.db -> table 'tweets').

Required env var:
  TWITTER_BEARER_TOKEN  - your Twitter API v2 bearer token

Run:
  export TWITTER_BEARER_TOKEN="your_token_here"
  python collector.py
"""

import os
import sqlite3
import time
from datetime import datetime
import json
import logging
from typing import Optional

import tweepy  # pip install tweepy
import pandas as pd
from transformers import pipeline  # pip install transformers
from tqdm import tqdm

# Suppress TensorFlow/transformers warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# -----------------------
# Configuration
# -----------------------
DB_PATH = "tweets.db"
TABLE_NAME = "tweets"
# keywords to track (change to whatever product/brand/event you want)
TRACK_KEYWORDS = ["tesla", "elon musk", "tesla stock", "Model 3"]  # example
# You may want to set more advanced matching rules in Twitter rules
SLEEP_ON_ERROR = 30  # seconds to wait on failure before reconnect

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------
# Initialize sentiment pipeline
# -----------------------
# Use distilbert finetuned for sentiment (fast-ish). On CPU it's okay for low volume.
print("Loading transformer sentiment pipeline (this may take a while the first time)...")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # CPU (set 0 if you have CUDA GPU)
)
print("✓ Sentiment pipeline loaded successfully.")

# -----------------------
# DB helpers
# -----------------------
def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
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

def insert_tweet(conn: sqlite3.Connection, tweet_record: dict):
    placeholders = ", ".join("?" * len(tweet_record))
    columns = ", ".join(tweet_record.keys())
    sql = f"INSERT OR IGNORE INTO {TABLE_NAME} ({columns}) VALUES ({placeholders})"
    cur = conn.cursor()
    cur.execute(sql, tuple(tweet_record.values()))
    conn.commit()

# -----------------------
# Twitter Streaming Client
# -----------------------
class MyStream(tweepy.StreamingClient):
    def __init__(self, bearer_token, db_conn, *args, **kwargs):
        super().__init__(bearer_token, *args, **kwargs)
        self.db_conn = db_conn

    def on_connect(self):
        logging.info("Connected to Twitter streaming endpoint.")

    def on_tweet(self, tweet):
        # This receives simple Tweet objects (no user expansions unless requested).
        try:
            text = tweet.text
            tweet_id = str(tweet.id)
            created_at = getattr(tweet, "created_at", datetime.utcnow().isoformat())
            
            # Basic classification
            res = sentiment_pipe(text[:512])  # limit length to avoid super long inputs
            label = res[0]["label"]  # e.g., 'POSITIVE' or 'NEGATIVE'
            score = float(res[0]["score"])

            # Basic username placeholder (if we expand users, we could get actual username)
            username = None

            record = {
                "id": tweet_id,
                "created_at": str(created_at),
                "username": username,
                "text": text,
                "sentiment_label": label,
                "sentiment_score": score,
                "collected_at": datetime.utcnow().isoformat(),
            }

            insert_tweet(self.db_conn, record)
            logging.info(f"Saved tweet {tweet_id} | {label} {score:.2f} | {text[:60]!r}")

        except Exception as e:
            logging.exception("Error processing tweet: %s", e)

    def on_errors(self, errors):
        logging.error("Stream error: %s", errors)

    def on_connection_error(self):
        logging.error("Connection error -- will attempt reconnect.")
        self.disconnect()

# -----------------------
# Setup streaming rules
# -----------------------
def setup_rules(client: tweepy.StreamingClient, keywords):
    # Delete existing rules first (optional)
    try:
        rules = client.get_rules()
        if rules and rules.data:
            rule_ids = [r.id for r in rules.data]
            client.delete_rules(rule_ids)
            logging.info("Deleted existing rules.")
    except Exception as e:
        logging.warning(f"Could not delete existing rules: {e}")

    # Add new rules based on keywords
    query = " OR ".join(f'"{k}"' for k in keywords)
    # Basic rule: no retweets, english only (modify as needed)
    rule_value = f"({query}) -is:retweet lang:en"
    client.add_rules(tweepy.StreamRule(value=rule_value, tag="keyword_rule"))
    logging.info(f"Added stream rule: {rule_value}")

# -----------------------
# Main function
# -----------------------
def main():
    # ✅ FIX: Get bearer token from environment variable
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    
    if not bearer:
        raise RuntimeError(
            "Set TWITTER_BEARER_TOKEN env var:\n"
            "  export TWITTER_BEARER_TOKEN='your_token_here'\n"
            "Or set it in your .env file"
        )

    logging.info("Bearer token found, initializing...")
    
    conn = init_db(DB_PATH)
    logging.info(f"Database initialized: {DB_PATH}")

    # Create stream client with expansions to get created_at etc. (simple usage)
    stream = MyStream(bearer, conn, wait_on_rate_limit=True)

    # set up rules
    setup_rules(stream, TRACK_KEYWORDS)

    # Start streaming (this will block)
    while True:
        try:
            logging.info("Starting filtered stream...")
            # Use expansions if you prefer more data (users, etc). For brevity, simple.
            stream.filter(tweet_fields=["created_at", "lang"], expansions=[])
        except KeyboardInterrupt:
            logging.info("Stopping stream (KeyboardInterrupt).")
            break
        except Exception as e:
            logging.exception("Streaming failed, sleeping then reconnecting: %s", e)
            time.sleep(SLEEP_ON_ERROR)
            continue
    
    conn.close()
    logging.info("Database connection closed.")

if __name__ == "__main__":
    main()