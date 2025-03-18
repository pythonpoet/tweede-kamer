#Imports
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
import nltk
import os
nltk.download('punkt')
nltk.download('punkt_tab')  # Download sentence tokenizer data
from nltk.tokenize import sent_tokenize

root_dir = "data/speeches2014-2024"

def translate(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def split_sentences(text, max_chunk_size=4000):

    if len(text) <= max_chunk_size:
        return [text]

    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0


    
    # Split into chunks while preserving sentence boundaries
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
    # Translate chunks sequentially
    translated_text = []
    for chunk in chunks:
        try:
            translated = translate(chunk)
            translated_text.append(translated)
        except Exception as e:
            print(f"Error translating chunk: {e}")
            translated_text.append("")  # Add empty string on failure
            
    return ' '.join(translated_text)

def safe_translate(text, max_text_lenght=4000):
    chunks = split_sentences(text, max_chunk_size=max_text_lenght)
    return split_translate(chunks)

if __name__ == "__main__":
    
    # Iterate over all files in the directory
    for filename in os.listdir(root_dir):
        # Construct the full file path
        file_path = os.path.join(root_dir, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Apply the translation function with a progress bar
        tqdm.pandas(desc=f"Translating {filename}")
        df['speech_text_google_translate'] = df['speech_text'].progress_apply(safe_translate)
        
        # Save the modified DataFrame back to the same file
        df.to_csv(file_path, index=False)
        print(f"Processed and saved: {filename}")