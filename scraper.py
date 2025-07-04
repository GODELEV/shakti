import os
import requests
import hashlib
from icrawler.builtin import BingImageCrawler
from duckduckgo_search import DDGS
from PIL import Image
from tqdm import tqdm
from utils import is_valid_image, hash_file, get_next_filename, resize_image

def download_with_bing(keyword, needed, image_dir, downloaded_hashes, resize_size, progress_callback=None):
    crawler = BingImageCrawler(storage={"root_dir": image_dir})
    before_files = set(os.listdir(image_dir))
    crawler.crawl(keyword=keyword, max_num=needed, min_size=(64,64))
    after_files = set(os.listdir(image_dir))
    new_files = list(after_files - before_files)
    added = 0
    for fname in new_files:
        path = os.path.join(image_dir, fname)
        if not is_valid_image(path):
            os.remove(path)
            continue
        h = hash_file(path)
        if h in downloaded_hashes:
            os.remove(path)
            continue
        downloaded_hashes.add(h)
        ext = os.path.splitext(fname)[1].lower()
        new_name = get_next_filename(image_dir, ext)
        new_path = os.path.join(image_dir, new_name)
        if path != new_path:
            os.rename(path, new_path)
        if resize_size is not None:
            resize_image(new_path, resize_size)
        added += 1
        if progress_callback:
            progress_callback(1)
    return added

def download_with_duckduckgo(keyword, needed, image_dir, downloaded_hashes, resize_size, progress_callback=None):
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
                fname = get_next_filename(image_dir, ext)
                path = os.path.join(image_dir, fname)
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
                        resize_image(path, resize_size)
                    added += 1
                    if progress_callback:
                        progress_callback(1)
                if added >= needed:
                    break
            except Exception:
                continue
    return added

def download_images(keywords, target_count, image_dir, resize_size, progress_callback=None):
    """
    Download images using Bing and DuckDuckGo until target_count is reached.
    Deduplicate, skip broken/corrupt images, and resize as needed.
    Returns: dict of {keyword: num_successful_images}
    """
    downloaded_hashes = set()
    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)
        if is_valid_image(path):
            downloaded_hashes.add(hash_file(path))
    count = len(downloaded_hashes)
    stats = {kw: 0 for kw in keywords}
    pbar = tqdm(total=target_count, initial=count, desc="Total Images")
    while count < target_count:
        needed = target_count - count
        for kw in keywords:
            if count >= target_count:
                break
            added = download_with_bing(kw, needed, image_dir, downloaded_hashes, resize_size, progress_callback=pbar.update)
            stats[kw] += added
            count += added
            if count >= target_count:
                break
        if count < target_count:
            for kw in keywords:
                if count >= target_count:
                    break
                added = download_with_duckduckgo(kw, needed, image_dir, downloaded_hashes, resize_size, progress_callback=pbar.update)
                stats[kw] += added
                count += added
                if count >= target_count:
                    break
        if all(stats[kw] == 0 for kw in keywords):
            print("No more images found for your keywords.")
            break
    pbar.close()
    return stats 