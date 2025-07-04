import os
import sys
import hashlib
import requests
import pandas as pd
from icrawler.builtin import BingImageCrawler
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from tqdm import tqdm
from duckduckgo_search import DDGS
import random
import time
from expansion import expand_keywords
from scraper import download_images
from captioner import generate_caption, load_blip_model
from exporter import export_to_json, write_summary
from utils import ensure_dirs, is_valid_image
import threading
import queue

OUTPUT_ROOT = 'output'
DATASET_DIR = os.path.join(OUTPUT_ROOT, 'dataset')
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
CAPTIONS_FILE = os.path.join(DATASET_DIR, 'captions.txt')
SUMMARY_FILE = os.path.join(DATASET_DIR, 'summary.txt')
JSON_FILE = os.path.join(DATASET_DIR, 'dataset.json')

RES_OPTIONS = {
    "1": (64, 64),
    "2": (128, 128),
    "3": (192, 192),
    "4": (256, 256),
    "5": (512, 512),
    "6": None
}

BATCH_SIZE = 300

# --- Keyword Expansion ---
def expand_keywords(prompt, n_variations=15):
    # Simple synonym/phrase expansion for demo; can be improved with NLP
    synonyms = {
        'river': ['stream', 'creek', 'brook', 'waterway'],
        'mountain': ['hill', 'peak', 'summit', 'ridge'],
        'forest': ['woods', 'jungle', 'grove', 'timberland'],
        'sky': ['heavens', 'atmosphere', 'clouds'],
    }
    adjectives = ['serene', 'rocky', 'lush', 'misty', 'sunny', 'tranquil', 'majestic', 'picturesque']
    base_words = prompt.lower().split()
    variations = set()
    for _ in range(n_variations):
        phrase = []
        for word in base_words:
            if word in synonyms and random.random() < 0.5:
                phrase.append(random.choice(synonyms[word]))
            else:
                phrase.append(word)
        if random.random() < 0.7:
            phrase = [random.choice(adjectives)] + phrase
        variations.add(' '.join(phrase))
    variations.add(prompt)
    return list(variations)[:n_variations]

# --- Utility Functions ---
def ensure_dirs(image_dir, clear_images=False):
    os.makedirs(image_dir, exist_ok=True)
    if clear_images:
        for fname in os.listdir(image_dir):
            file_path = os.path.join(image_dir, fname)
            if os.path.isfile(file_path):
                os.remove(file_path)

def hash_file(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def deduplicate_images(image_dir):
    hashes = {}
    for fname in sorted(os.listdir(image_dir)):
        path = os.path.join(image_dir, fname)
        if not is_valid_image(path):
            os.remove(path)
            continue
        h = hash_file(path)
        if h in hashes:
            os.remove(path)
        else:
            hashes[h] = fname

def get_next_filename(image_dir, ext):
    existing = [int(os.path.splitext(f)[0]) for f in os.listdir(image_dir) if f.endswith(ext) and os.path.splitext(f)[0].isdigit()]
    next_num = max(existing, default=0) + 1
    return f"{next_num:06d}{ext}"

# --- Downloaders ---
def download_with_bing(keyword, needed, downloaded_hashes, resize_size):
    crawler = BingImageCrawler(storage={"root_dir": IMAGES_DIR})
    before_files = set(os.listdir(IMAGES_DIR))
    crawler.crawl(keyword=keyword, max_num=needed, min_size=(64,64))
    after_files = set(os.listdir(IMAGES_DIR))
    new_files = list(after_files - before_files)
    added = 0
    for fname in new_files:
        path = os.path.join(IMAGES_DIR, fname)
        if not is_valid_image(path):
            os.remove(path)
            continue
        h = hash_file(path)
        if h in downloaded_hashes:
            os.remove(path)
            continue
        downloaded_hashes.add(h)
        ext = os.path.splitext(fname)[1].lower()
        # Use next available filename
        new_name = get_next_filename(IMAGES_DIR, ext)
        new_path = os.path.join(IMAGES_DIR, new_name)
        if path != new_path:
            os.rename(path, new_path)
        if resize_size is not None:
            try:
                img = Image.open(new_path)
                img = img.resize(resize_size, Image.LANCZOS)
                img.save(new_path)
            except Exception as e:
                print(f"Failed to resize {new_name}: {e}")
        added += 1
    return added

def download_with_duckduckgo(keyword, needed, downloaded_hashes, resize_size):
    added = 0
    with DDGS() as ddgs:
        results = ddgs.images(keywords=keyword, max_results=needed)
        for result in tqdm(results, desc=f"DuckDuckGo: {keyword}", total=needed):
            url = result.get('image')
            if not url:
                continue
            try:
                ext = os.path.splitext(url)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    ext = '.jpg'
                # Use next available filename
                fname = get_next_filename(IMAGES_DIR, ext)
                path = os.path.join(IMAGES_DIR, fname)
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(path, 'wb') as f:
                        f.write(r.content)
                    if not is_valid_image(path):
                        os.remove(path)
                        continue
                    h = hash_file(path)
                    if h in downloaded_hashes:
                        os.remove(path)
                        continue
                    downloaded_hashes.add(h)
                    if resize_size is not None:
                        try:
                            img = Image.open(path)
                            img = img.resize(resize_size, Image.LANCZOS)
                            img.save(path)
                        except Exception as e:
                            print(f"Failed to resize {fname}: {e}")
                    added += 1
                if added >= needed:
                    break
            except Exception:
                continue
    return added

# --- Main Download Loop ---
def download_images(keywords, target_count, image_dir, resize_size, progress_callback=None):
    downloaded_hashes = set()
    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)
        if is_valid_image(path):
            downloaded_hashes.add(hash_file(path))
    count = len(downloaded_hashes)
    pbar = tqdm(total=target_count, initial=count, desc="Total Images")
    while count < target_count:
        needed = target_count - count
        for kw in keywords:
            if count >= target_count:
                break
            # Try Bing first
            added = download_with_bing(kw, needed, downloaded_hashes, resize_size)
            count += added
            pbar.update(added)
            if count >= target_count:
                break
        # If still not enough, try DuckDuckGo
        if count < target_count:
            for kw in keywords:
                if count >= target_count:
                    break
                added = download_with_duckduckgo(kw, needed, downloaded_hashes, resize_size)
                count += added
                pbar.update(added)
                if count >= target_count:
                    break
        if added == 0:
            print("No more images found for your keywords.")
            break
    pbar.close()
    deduplicate_images(image_dir)
    print(f"Downloaded {count} valid images.")
    if count < target_count:
        print(f"Warning: Only {count} images could be downloaded for your keywords.")
    return {kw: added for kw in keywords}

# --- Resize Utility (if needed for all images) ---
def resize_images(image_dir, resolution):
    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)
        if is_valid_image(path):
            try:
                img = Image.open(path)
                img = img.resize(resolution, Image.LANCZOS)
                img.save(path)
            except Exception as e:
                print(f"Failed to resize {fname}: {e}")

# --- Caption Generation ---
def generate_captions_file():
    processor, model, device = load_blip_model()
    data = []
    for fname in sorted(os.listdir(IMAGES_DIR)):
        img_path = os.path.join(IMAGES_DIR, fname)
        try:
            caption = generate_caption(img_path, processor, model, device)
        except Exception as e:
            print(f"Failed to caption {fname}: {e}")
            caption = "No caption available."
        data.append({'image': fname, 'caption': caption})
    df = pd.DataFrame(data)
    df.to_csv(CAPTIONS_FILE, sep='\t', index=False, header=False)
    print(f"Captions saved to {CAPTIONS_FILE}")

# --- CLI ---
def get_user_input():
    prompt = input('Enter your dataset topic or prompt: ').strip()
    while True:
        try:
            num_images = int(input('Enter number of images to download: '))
            if num_images > 0:
                break
            else:
                print('Please enter a positive integer.')
        except ValueError:
            print('Invalid input. Please enter a number.')
    print('Choose image resolution:')
    print('1. 64x64')
    print('2. 128x128')
    print('3. 192x192')
    print('4. 256x256')
    print('5. 512x512')
    print('6. No resize')
    while True:
        res_choice = input('Enter option number (1-6): ').strip()
        if res_choice in RES_OPTIONS:
            resize_size = RES_OPTIONS[res_choice]
            break
        else:
            print('Invalid option. Please enter a number between 1 and 6.')
    return prompt, num_images, resize_size

# --- Main ---
def main():
    print("\n==== Shakti EPIC: Universal Smart Dataset Generator ====")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    ensure_dirs(image_dir=IMAGES_DIR, clear_images=True)
    prompt, num_images, resize_size = get_user_input()
    print("\nExpanding keywords...")
    keywords = expand_keywords(prompt, n_variations=18)
    print(f"Keyword variations: {keywords}")
    print("\nStarting image download and captioning (in parallel)...")
    t0 = time.time()

    # Prepare captioning thread
    img_queue = queue.Queue()
    caption_dict = {}
    stop_event = threading.Event()
    processor, model, device = load_blip_model()
    gpu_status = (device == "cuda")
    caption_thread = threading.Thread(target=threaded_caption_worker, args=(img_queue, caption_dict, stop_event, processor, model, device))
    caption_thread.start()

    # Download images in batches, prompt user after each batch
    stats = {kw: 0 for kw in keywords}
    downloaded = set()
    total_downloaded = 0
    batch_num = 0
    for batch_start in range(0, num_images, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_images)
        batch_target = batch_end - batch_start
        print(f"\nDownloading images {batch_start+1} to {batch_end}...")
        batch_stats = download_images(keywords, batch_target, IMAGES_DIR, resize_size)
        for k, v in batch_stats.items():
            stats[k] += v
        # Enqueue new images for captioning
        new_files = [f for f in sorted(os.listdir(IMAGES_DIR)) if is_valid_image(os.path.join(IMAGES_DIR, f)) and f not in downloaded]
        for fname in new_files:
            img_queue.put(fname)
            downloaded.add(fname)
        total_downloaded = len(downloaded)
        print(f"Downloaded {total_downloaded} images so far.")
        # Prompt user after each batch except the last
        if total_downloaded < num_images:
            while True:
                user_choice = input(f"\n{total_downloaded} images downloaded. Do you want to stop and make a dataset now? (yes/no): ").strip().lower()
                if user_choice in ("yes", "no", "y", "n"):
                    break
                else:
                    print("Please enter yes or no.")
            if user_choice.startswith('y'):
                print("Stopping download. Finalizing dataset...")
                break
    # Signal caption thread to finish
    stop_event.set()
    img_queue.join()
    caption_thread.join()
    t1 = time.time()
    # Write captions.txt
    data = [{'image': fname, 'caption': caption_dict.get(fname, "No caption available.")} for fname in sorted(downloaded)]
    pd.DataFrame(data).to_csv(CAPTIONS_FILE, sep='\t', index=False, header=False)
    print(f"Captions saved to {CAPTIONS_FILE}")
    print("\nWriting summary...")
    write_summary(SUMMARY_FILE, {
        'prompt': prompt,
        'keywords': keywords,
        'stats': stats,
        'total_images': total_downloaded,
        'resolution': f"{resize_size[0]}x{resize_size[1]}" if resize_size else 'original',
        'time': t1 - t0,
        'gpu_status': gpu_status,
        'author': 'Akshit'
    })
    # Optional JSON export
    while True:
        export_json = input("\nDo you want to export your dataset to Hugging Face-style JSON? (yes/no): ").strip().lower()
        if export_json in ('yes', 'no', 'y', 'n'):
            break
        else:
            print('Please enter yes or no.')
    if export_json.startswith('y'):
        export_to_json(CAPTIONS_FILE, JSON_FILE)
    print(f"\nAll done! Your dataset is ready in the '{DATASET_DIR}/' folder (inside 'output/').")

def threaded_caption_worker(img_queue, caption_dict, stop_event, processor, model, device):
    while not stop_event.is_set() or not img_queue.empty():
        try:
            fname = img_queue.get(timeout=1)
        except queue.Empty:
            continue
        img_path = os.path.join(IMAGES_DIR, fname)
        try:
            caption = generate_caption(img_path, processor, model, device)
        except Exception as e:
            print(f"Failed to caption {fname}: {e}")
            caption = "No caption available."
        caption_dict[fname] = caption
        img_queue.task_done()

if __name__ == "__main__":
    main() 