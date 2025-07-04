import random

# Some generic adjectives and templates for prompt engineering
ADJECTIVES = [
    'beautiful', 'serene', 'futuristic', 'ancient', 'vintage', 'modern', 'colorful', 'minimalist',
    'majestic', 'mysterious', 'realistic', 'abstract', 'dynamic', 'classic', 'elegant', 'epic', 'iconic',
    'urban', 'natural', 'fantasy', 'retro', 'luxury', 'wild', 'rare', 'unique', 'detailed', 'high quality'
]
TEMPLATES = [
    '{adj} {prompt}',
    '{prompt} in nature',
    '{prompt} at night',
    '{prompt} in the city',
    '{prompt} landscape',
    '{adj} {prompt} scene',
    '{prompt} with {adj} background',
    '{prompt} style',
    '{prompt} aesthetic',
    '{adj} {prompt} art',
    '{prompt} photo',
    '{prompt} wallpaper',
    '{prompt} illustration',
    '{prompt} design',
    '{prompt} concept',
    '{prompt} closeup',
    '{prompt} portrait',
    '{prompt} macro',
    '{prompt} view',
    '{prompt} 4k',
]

# Some basic synonym mapping for common words (expandable)
SYNONYMS = {
    'river': ['stream', 'creek', 'brook', 'waterway'],
    'mountain': ['hill', 'peak', 'summit', 'ridge', 'alps'],
    'forest': ['woods', 'jungle', 'grove', 'timberland'],
    'car': ['automobile', 'vehicle', 'sedan', 'coupe', 'convertible'],
    'robot': ['android', 'automaton', 'machine', 'cyborg'],
    'animal': ['creature', 'beast', 'fauna', 'wildlife'],
    'temple': ['shrine', 'sanctuary', 'pagoda', 'church'],
    'city': ['metropolis', 'urban area', 'town', 'capital'],
    'fashion': ['style', 'trend', 'couture', 'apparel'],
    'sky': ['heavens', 'atmosphere', 'clouds'],
    'flower': ['blossom', 'bloom', 'petal', 'flora'],
    'tree': ['oak', 'pine', 'willow', 'maple', 'birch'],
    'cat': ['kitten', 'feline', 'tomcat'],
    'dog': ['puppy', 'canine', 'hound', 'pooch'],
    'building': ['structure', 'skyscraper', 'tower', 'edifice'],
    'person': ['human', 'individual', 'figure', 'character'],
    'portrait': ['headshot', 'profile', 'likeness'],
    'painting': ['artwork', 'canvas', 'illustration'],
    'vintage': ['retro', 'classic', 'old-fashioned'],
    'cyberpunk': ['futuristic', 'sci-fi', 'techno', 'dystopian'],
}

def expand_keywords(prompt, n_variations=15):
    """
    Expand a user prompt into a list of 10-20 semantically related keyword variations.
    Uses synonyms, adjectives, and prompt templates. Works for any domain.
    """
    prompt_words = prompt.lower().split()
    # Replace words with synonyms randomly
    def synonymize(words):
        return [random.choice(SYNONYMS[w]) if w in SYNONYMS and random.random() < 0.5 else w for w in words]
    variations = set()
    for _ in range(n_variations * 2):  # try more, filter later
        words = synonymize(prompt_words)
        base = ' '.join(words)
        adj = random.choice(ADJECTIVES)
        template = random.choice(TEMPLATES)
        phrase = template.format(prompt=base, adj=adj)
        variations.add(phrase)
        # Also add just the synonymized base
        variations.add(base)
    # Always include the original prompt
    variations.add(prompt)
    # Return up to n_variations unique prompts
    return list(variations)[:n_variations] 