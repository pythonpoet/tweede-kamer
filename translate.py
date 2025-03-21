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


# Proxy List URL (Replace with actual proxy list URL)
#PROXY_LIST_URL = "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/http/data.txt"
PROXY_LIST_URL = "https://freeproxydb.com/api/proxy/search?country=&protocol=socks5&anonymity=&speed=0,60&https=0&page_index=1&page_size=10"

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


async def async_translate_bulk(text_list, proxy=None):
    """Translates a list of texts using GoogleTranslator in bulk with retries."""
    if not text_list:
        return []

    retries = 0
    while retries < MAX_RETRIES:
        try:
            logging.info(f"Translating batch with proxy: {proxy} (Attempt {retries+1})")
            translator = GoogleTranslator(
                source="auto", target="en", proxies={"http": proxy, "https": proxy}
            )

            start_time = time.time()

            # Run synchronous translation in a separate thread
            translations = await asyncio.gather(
                *(asyncio.to_thread(translator.translate, text) for text in text_list)
            )

            end_time = time.time()
            logging.info(f"Translated {len(text_list)} texts in {end_time - start_time:.2f} seconds.")

            return translations
        except Exception as e:
            error_message = str(e)
            logging.warning(f"Translation failed with proxy {proxy}: {error_message}")

            # Rate limit handling
            if "too many requests" in error_message.lower():
                wait_time = (2 ** retries) * REQUEST_DELAY  # Exponential backoff
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                proxy = get_proxy()  # Switch to a new proxy

            retries += 1

    logging.error("Failed to translate after multiple retries.")
    return [""] * len(text_list)  # Return empty translations for failed attempts

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
        proxy = get_proxy()

        translations = await async_translate_bulk(batch_texts, proxy)
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
            df = df.head(300)

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
            results.to_csv("small_sample.csv", index=False)
            logging.info(f"Successfully processed and saved: {filename}")

        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")


if __name__ == "__main__":
    print("Starting translation process... Check translation_debug.log for details.")
    fetch_proxies()  # Fetch proxies once at the beginning
    asyncio.run(process_all_files())
