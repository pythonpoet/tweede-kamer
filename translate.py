import requests
import pandas as pd
import time
import random
import logging
from deep_translator import GoogleTranslator
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np  # Added for NaN checking

# Setup logging
logging.basicConfig(filename="translation_errors.log", level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# Proxy List URL
PROXY_LIST_URL = "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/http/data.txt"  # Replace with actual proxy URL

# Google API Limits
REQUESTS_PER_SECOND = 5
MAX_RETRIES = 5
MAX_WORKERS = 3  # Keep requests under limit
REQUEST_DELAY = 1 / REQUESTS_PER_SECOND  # Ensure we don't exceed 5 requests/sec

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

def translate(text, proxy=None):
    """Translates text using GoogleTranslator with automatic retries and rate limit handling."""
    if not text or not isinstance(text, str) or text.strip() == "":
        return ""

    retries = 0
    while retries < MAX_RETRIES:
        try:
            time.sleep(REQUEST_DELAY)  # Enforce rate limit
            proxies = {"http": proxy, "https": proxy} if proxy else None
            translated = GoogleTranslator(source='auto', target='en').translate(text, proxies=proxies)
            if translated:
                return translated
        except Exception as e:
            error_message = str(e)
            logging.warning(f"Translation failed with proxy {proxy}: {error_message}")

            # If rate limit is exceeded, wait and retry
            if "too many requests" in error_message.lower():
                wait_time = (2 ** retries) * REQUEST_DELAY  # Exponential backoff
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                proxy = get_new_proxy()  # Get a fresh proxy

            retries += 1

    logging.error(f"Failed to translate after {MAX_RETRIES} retries.")
    return ""

def translate_dataframe_parallel(df, text_column, translated_column):
    """Translates only missing values in a Pandas DataFrame in parallel."""
    tqdm.pandas(desc="Translating speeches")

    # Find indices where translation is missing (NaN)
    missing_indices = df[df[translated_column].isna()].index.tolist()
    
    if not missing_indices:
        print("No missing translations. Skipping translation process.")
        return df[translated_column]  # Return the same column unchanged

    results = df[translated_column].copy()  # Keep existing translations

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {executor.submit(translate, df.loc[idx, text_column], get_new_proxy()): idx for idx in missing_indices}

        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing translations"):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logging.error(f"Error in parallel translation: {e}")
                results[idx] = ""

    return results

if __name__ == "__main__":
    root_dir = "data/speeches2014-2024"

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
            df['speech_text_google_translate'] = translate_dataframe_parallel(df, 'speech_text', 'speech_text_google_translate')

            df.to_csv(file_path, index=False)
            print(f"Processed and saved: {filename}")

        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")
