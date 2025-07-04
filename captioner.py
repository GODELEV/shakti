import os
import time
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration

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

def generate_captions(image_dir, captions_file, use_tqdm=True):
    """
    Generate BLIP captions for all images in image_dir and write to captions_file (tab-separated).
    Returns: (gpu_status, time_taken)
    """
    processor, model, device = load_blip_model()
    gpu_status = (device == "cuda")
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    data = []
    start = time.time()
    iterator = tqdm(image_files, desc="Captioning Images") if use_tqdm else image_files
    for fname in iterator:
        img_path = os.path.join(image_dir, fname)
        try:
            caption = generate_caption(img_path, processor, model, device)
        except Exception as e:
            print(f"Failed to caption {fname}: {e}")
            caption = "No caption available."
        data.append({'image': fname, 'caption': caption})
    df = pd.DataFrame(data)
    df.to_csv(captions_file, sep='\t', index=False, header=False)
    elapsed = time.time() - start
    print(f"Captions saved to {captions_file}")
    return gpu_status, elapsed 