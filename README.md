# Shakti 🧠📸

**Shakti** is an automatic image dataset generator for AI/ML projects. You provide a **keyword** and the number of images, and Shakti:

- Downloads images from Google
- Generates captions using a pretrained BLIP model
- Saves everything in a clean format for training

## 📦 Usage

```bash
python shakti.py
```

## Features
- 🔍 Ask the user for keyword and number of images
- 📥 Download images from Google using icrawler
- 📝 Generate captions.txt file (using BLIP)
- 📁 Output structured in clean dataset folder
- 📘 Includes README.md and LICENSE files
- ✅ Easy to expand for training

## Output Structure
```
dataset/
  images/
    000001.jpg
    ...
  captions.txt  # image_name\tcaption
```

## Installation

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python shakti.py
   ```

## Requirements
- Python 3.7+
- torch
- transformers
- pillow
- icrawler
- pandas

## License
See [LICENSE](LICENSE). 