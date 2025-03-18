import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
import nltk
import os
import time
import logging

# Setup logging
logging.basicConfig(filename="translation_errors.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Download necessary NLTK data
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

root_dir = "data/speeches2014-2024"
REQUESTS_PER_MINUTE = 300  # Adjust based on API limits
REQUEST_DELAY = 60 / REQUESTS_PER_MINUTE  # Time delay between requests
MAX_RETRIES = 5  # Maximum retries on failure

def translate(text):
    """Translate text with automatic retries if rate limit is exceeded."""
    if not text or not isinstance(text, str):
        return ""

    retries = 0
    while retries < MAX_RETRIES:
        try:
            time.sleep(REQUEST_DELAY)  # Enforce rate limit
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            
            if translated:
                return translated  # Successful translation
            
        except Exception as e:
            error_message = str(e)
            if "Rate Limit Exceeded" in error_message or "quota" in error_message:
                wait_time = (2 ** retries) * REQUEST_DELAY  # Exponential backoff
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                logging.error(f"Translation error: {error_message} | Text: {text[:50]}")
                break  # Stop retrying for non-quota errors

    return ""  # Return empty string if all retries fail

def split_sentences(text, max_chunk_size=4000):
    """Split text into smaller chunks while preserving sentence boundaries."""
    if not text or not isinstance(text, str):
        return []
    
    if len(text) <= max_chunk_size:
        return [text]

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def split_translate(chunks):
    """Translate each chunk sequentially with retry handling."""
    translated_text = []
    for chunk in chunks:
        translated = translate(chunk)
        translated_text.append(translated)
    return ' '.join(translated_text)

def safe_translate(text, max_text_length=4000):
    """Ensure translation handles errors properly."""
    chunks = split_sentences(text, max_chunk_size=max_text_length)
    if not chunks:
        return ""
    return split_translate(chunks)

if __name__ == "__main__":
    for filename in os.listdir(root_dir):
        file_path = os.path.join(root_dir, filename)

        try:
            df = pd.read_csv(file_path)

            if 'speech_text' not in df.columns:
                logging.error(f"Missing 'speech_text' column in {filename}")
                continue

            tqdm.pandas(desc=f"Translating {filename}")
            df['speech_text_google_translate'] = df['speech_text'].progress_apply(safe_translate)

            df.to_csv(file_path, index=False)
            print(f"Processed and saved: {filename}")

        except Exception as e:
            logging.error(f"Error processing file {filename}: {e}")
