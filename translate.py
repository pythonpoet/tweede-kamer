import asyncio
import requests
import pandas as pd
import time
import random
import logging
import numpy as np
import os
from tqdm.asyncio import tqdm_asyncio
from deep_translator import GoogleTranslator
import json
from urllib.parse import urlparse
#from fake_useragent import UserAgent
import random

# Use these domains randomly
TRANSLATE_DOMAINS = [
    "translate.google.com",  # Default
    "translate.google.co.uk",
    "translate.google.de",
    "translate.google.fr",
    "translate.google.es",
    "translate.google.ru",
    "translate.google.cn",
    "translate.google.jp"
]


# Proxy List URL (Replace with actual proxy list URL)
#PROXY_LIST_URL = "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/http/data.txt"
PROXY_LIST_URL = "https://freeproxydb.com/api/proxy/search?country=&protocol=socks5&anonymity=&speed=0,60&https=0&page_index=1&page_size=200"

# Logging setup
logging.basicConfig(
    filename="translation_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Google API Limits
REQUESTS_PER_SECOND = 10
BATCH_SIZE = 10
MAX_RETRIES = 5
REQUEST_DELAY = 1 / REQUESTS_PER_SECOND
TRANSLATION_TIMEOUT = 15

# Store used proxies
proxies = []
current_proxy_index = 0


def fetch_proxies():
    """Fetches proxies once at the start to avoid unnecessary requests."""
    global proxies
    try:
        logging.info("Fetching new proxies...")
        response = requests.get(PROXY_LIST_URL, timeout=10)
        response.raise_for_status()
        proxies = [
            entry["connect_string"].strip()
            for entry in response.json()["data"]["data"]
            if "connect_string" in entry
        ]
        logging.info(f"Fetched {len(proxies)} proxies.")
    except requests.RequestException as e:
        logging.error(f"Failed to fetch proxies: {e}")
        proxies = []


def get_proxy():
    """Cycles through available proxies instead of fetching a new list each time."""
    global current_proxy_index, proxies
    if not proxies:
        fetch_proxies()

    if proxies:
        proxy = proxies[current_proxy_index % len(proxies)]
        current_proxy_index += 1  # Move to the next proxy in the list
        logging.info(f"Using proxy: {proxy}")

        return proxy
    else:
        logging.warning("No proxies available, using direct connection.")
        return None
def blacklist_proxy(proxy):
    """Temporarily remove bad proxies from rotation"""
    global proxies
    proxies = [p for p in proxies if p != proxy]

async def async_translate_bulk(text_list):
    """Enhanced translation function with multiple anti-ban features"""
    results = [""] * len(text_list)
    remaining = list(enumerate(text_list))
    # Randomly select a domain for each attempt
    selected_domain = random.choice(TRANSLATE_DOMAINS)

    for attempt in range(MAX_RETRIES):
        if not remaining:
            break

        proxy = get_proxy()
        translator = GoogleTranslator(
            source="auto",
            target="en",
            proxies={"http": proxy, "https": proxy},
            # Add browser-like headers
            # headers={
            #     "User-Agent": UserAgent().random,
            #     "Accept-Language": "en-US,en;q=0.9",
            #     "Referer": "https://translate.google.com/"
            # },
            service_url=selected_domain
        )

        try:
            # Randomize request timing
            jitter = random.uniform(0.5, 1.5)
            await asyncio.sleep(REQUEST_DELAY * jitter)

            # Split batch to smaller chunks
            chunk_size = random.randint(3, 5)
            for i in range(0, len(remaining), chunk_size):
                chunk = remaining[i:i+chunk_size]
                texts = [t[1] for t in chunk]

                translations = await asyncio.gather(
                    *(asyncio.to_thread(translator.translate, text)
                    for text in texts),
                    return_exceptions=True
                )

                # Process results
                new_remaining = []
                for (idx, orig), translation in zip(chunk, translations):
                    if isinstance(translation, Exception):
                        new_remaining.append((idx, orig))
                    elif translation.strip():
                        results[idx] = translation
                    else:
                        new_remaining.append((idx, orig))

                remaining = new_remaining
                if not remaining:
                    break

                # Random delay between chunks
                await asyncio.sleep(random.uniform(1, 3))

        except Exception as e:
            logging.error(f"Proxy {proxy} failed: {str(e)}")
            blacklist_proxy(proxy)
            continue

    return results

async def process_dataframe_async(df, text_column, translated_column):
    """Processes missing translations in batches asynchronously."""
    missing_indices = df[df[translated_column].isna()].index.tolist()
    if not missing_indices:
        logging.info("No missing translations, skipping translation process.")
        return df[translated_column]

    results = df[translated_column].copy()

    # Create progress bar outside the processing loop
    pbar = tqdm_asyncio(total=len(missing_indices), desc="Translating speeches", unit="rows")

    async def process_batch(batch_start):
        batch_end = batch_start + BATCH_SIZE
        batch_indices = missing_indices[batch_start:batch_end]
        batch_texts = df.loc[batch_indices, text_column].tolist()


        translations = await async_translate_bulk(batch_texts)
        return batch_start, translations

    tasks = [process_batch(i) for i in range(0, len(missing_indices), BATCH_SIZE)]

    for future in asyncio.as_completed(tasks):
        batch_start, translations = await future
        for offset, translation in enumerate(translations):
            results_idx = missing_indices[batch_start + offset]
            results[results_idx] = translation
        pbar.update(len(translations))

    pbar.close()
    return results

async def process_all_files():
    """Processes all speech files asynchronously."""
    root_dir = "data/speeches2014-2024"

    for filename in ["speeches_2015-2016.csv"]:#os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)

        try:
            logging.info(f"Processing file: {filename}")
            df = pd.read_csv(file_path)


            if "speech_text" not in df.columns:
                logging.error(f"Missing 'speech_text' column in {filename}")
                continue

            if "speech_text_google_translate" not in df.columns:
                df["speech_text_google_translate"] = np.nan

            results = await process_dataframe_async(
                df, "speech_text", "speech_text_google_translate"
            )
            df["speech_text_google_translate"] = results.copy()
            # veryfy
            if df["speech_text_google_translate"].equals(results):
                print("Assignment worked correctly.")
            else:
                print("Something went wrong with the assignment.")
            df.to_csv("small_sample.csv", index=False)
            logging.info(f"Successfully processed and saved: {filename}")

        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")


if __name__ == "__main__":
    print("Starting translation process... Check translation_debug.log for details.")
    fetch_proxies()  # Fetch proxies once at the beginning
    asyncio.run(process_all_files())
