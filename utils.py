import os
import hashlib
from PIL import Image
import PIL.Image as pil_image

def ensure_dirs(image_dir, clear_images=False):
    os.makedirs(image_dir, exist_ok=True)
    if clear_images:
        for fname in os.listdir(image_dir):
            file_path = os.path.join(image_dir, fname)
            if os.path.isfile(file_path):
                os.remove(file_path)

def is_valid_image(file_path):
    """Check if a file is a valid jpg/png image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.jpg', '.jpeg', '.png']
    except Exception:
        return False

def hash_file(file_path):
    """Return a hash of the file contents for duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_next_filename(image_dir, ext):
    """Get the next available 6-digit filename with the given extension in the directory."""
    existing = [int(os.path.splitext(f)[0]) for f in os.listdir(image_dir)
                if f.endswith(ext) and os.path.splitext(f)[0].isdigit()]
    next_num = max(existing, default=0) + 1
    return f"{next_num:06d}{ext}"

def resize_image(image_path, size):
    """Resize an image in-place to the given (width, height) tuple using LANCZOS filter."""
    try:
        img = Image.open(image_path)
        # Use LANCZOS from Resampling if available, else fallback to BICUBIC or 3
        resample = getattr(getattr(pil_image, 'Resampling', pil_image), 'LANCZOS',
                          getattr(pil_image, 'BICUBIC', 3))
        img = img.resize(size, resample)
        img.save(image_path)
    except Exception as e:
        print(f"Failed to resize {image_path}: {e}") 