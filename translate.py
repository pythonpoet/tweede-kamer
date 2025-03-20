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

# proxyfiy api = 8oEq6EJrQZzGKgMEQTzfaMNc4hgMLaXd6EqNRi5eXGmA
# https://proxifly.dev/

# Setup logging
logging.basicConfig(filename="translation_errors.log", level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Proxy List URL (Replace with actual proxy list URL)
PROXY_LIST_URL = "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/http/data.txt"

# Google API Limits
REQUESTS_PER_SECOND = 5
BATCH_SIZE = 10  # Number of translations per batch
MAX_RETRIES = 5
REQUEST_DELAY = 1 / REQUESTS_PER_SECOND  # Ensures we don't exceed 5 requests/sec

# Store used proxies
used_proxies = set()

def fetch_proxies():
    """Fetches the latest proxies from the website."""
    try:
        response = requests.get(PROXY_LIST_URL, timeout=10)
        response.raise_for_status()
        proxies = response.text.split("\n")
        return [proxy.strip() for proxy in proxies if proxy.strip()]
    except requests.RequestException as e:
        logging.error(f"Failed to fetch proxies: {e}")
        return []

def get_new_proxy():
    """Fetches a new proxy that hasn't been used recently."""
    global used_proxies
    proxies = fetch_proxies()
    
    if not proxies:
        logging.error("No proxies available!")
        return None
    
    new_proxies = [proxy for proxy in proxies if proxy not in used_proxies]
    
    if not new_proxies:  # If all proxies are used, reset the set
        used_proxies.clear()
        new_proxies = proxies

    proxy = random.choice(new_proxies)  # Pick a random new proxy
    used_proxies.add(proxy)  # Mark it as used
    return proxy

async def async_translate_bulk(text_list, proxy=None):
    """Translates a list of texts using GoogleTranslator in bulk with automatic retries."""
    if not text_list:
        return []

    retries = 0
    while retries < MAX_RETRIES:
        try:
            translator = GoogleTranslator(source="auto", target="en", proxies={"http": proxy, "https": proxy} if proxy else None)
            translations = [translator.translate(text) for text in text_list]  # Bulk translation
            return translations
        except Exception as e:
            error_message = str(e)
            logging.warning(f"Bulk translation failed with proxy {proxy}: {error_message}")

            # If rate limit is exceeded, wait and retry
            if "too many requests" in error_message.lower():
                wait_time = (2 ** retries) * REQUEST_DELAY  # Exponential backoff
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                proxy = get_new_proxy()  # Get a fresh proxy

            retries += 1

    logging.error("Failed to translate bulk data after retries.")
    return [""] * len(text_list)  # Return empty translations for failed attempts

async def process_dataframe_async(df, text_column, translated_column):
    """Processes missing translations in batches asynchronously."""
    tqdm_bar = tqdm_asyncio(total=df.shape[0], desc="Translating speeches", unit="rows")

    # Find indices where translation is missing (NaN)
    missing_indices = df[df[translated_column].isna()].index.tolist()
    
    if not missing_indices:
        print("No missing translations. Skipping translation process.")
        return df[translated_column]  # Return unchanged

    results = df[translated_column].copy()  # Keep existing translations
    tasks = []

    # Process in batches
    for i in range(0, len(missing_indices), BATCH_SIZE):
        batch_indices = missing_indices[i:i + BATCH_SIZE]
        batch_texts = df.loc[batch_indices, text_column].tolist()
        proxy = get_new_proxy()  # Get a fresh proxy for each batch

        tasks.append(async_translate_bulk(batch_texts, proxy))

    batch_results = await asyncio.gather(*tasks)  # Run all translation tasks

    # Update results in DataFrame
    for batch, batch_indices in zip(batch_results, missing_indices):
        for idx, translation in zip(batch_indices, batch):
            results[idx] = translation
            tqdm_bar.update(1)  # Update progress bar

    tqdm_bar.close()
    return results

async def process_all_files():
    root_dir = "data/speeches2014-2024"
    tasks = []

    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)

        try:
            df = pd.read_csv(file_path)

            if 'speech_text' not in df.columns:
                logging.error(f"Missing 'speech_text' column in {filename}")
                continue

            if 'speech_text_google_translate' not in df.columns:
                df['speech_text_google_translate'] = np.nan  # Initialize column if missing

            # Only translate NaN values
            df['speech_text_google_translate'] = await process_dataframe_async(df, 'speech_text', 'speech_text_google_translate')

            df.to_csv(file_path, index=False)
            print(f"Processed and saved: {filename}")

        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")

if __name__ == "__main__":
    asyncio.run(process_all_files())  # Run entire process asynchronously
