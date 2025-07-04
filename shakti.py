import os
import sys
import pandas as pd
from icrawler.builtin import GoogleImageCrawler
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

DATASET_DIR = 'dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
CAPTIONS_FILE = os.path.join(DATASET_DIR, 'captions.txt')


def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)


def get_user_input():
    keyword = input('Enter keyword to search images for: ').strip()
    while True:
        try:
            num_images = int(input('Enter number of images to download: '))
            if num_images > 0:
                break
            else:
                print('Please enter a positive integer.')
        except ValueError:
            print('Invalid input. Please enter a number.')
    return keyword, num_images


def download_images(keyword, num_images):
    print(f"Downloading {num_images} images for '{keyword}'...")
    crawler = GoogleImageCrawler(storage={"root_dir": IMAGES_DIR})
    crawler.crawl(keyword=keyword, max_num=num_images, min_size=(128,128), file_idx_offset=0)
    # Rename images to 6-digit zero-padded
    for idx, fname in enumerate(sorted(os.listdir(IMAGES_DIR))):
        ext = os.path.splitext(fname)[1].lower()
        new_name = f"{idx+1:06d}{ext}"
        os.rename(os.path.join(IMAGES_DIR, fname), os.path.join(IMAGES_DIR, new_name))


def load_blip_model():
    print("Loading BLIP model (may take a while the first time)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device


def generate_caption(image_path, processor, model, device):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


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


def main():
    ensure_dirs()
    keyword, num_images = get_user_input()
    download_images(keyword, num_images)
    generate_captions_file()
    print("\nAll done! Dataset is ready in 'dataset/' folder.")

if __name__ == "__main__":
    main() 