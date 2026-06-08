# src/rules.py
import re

# 27-ish coarse GoEmotions-style labels mapped to keywords.
# You can extend keywords for your dataset.
RULE_KEYWORDS = {
    "admiration": ["admire", "admiration", "respect", "impressed"],
    "amusement": ["lol", "haha", "lmao", "funny", "hilarious", "amused"],
    "anger": ["angry", "furious", "rage", "pissed", "annoyed", "annoyance"],
    "annoyance": ["annoyed", "irritat", "ugh", "bothered"],
    "anxiety": ["anxious", "anxiety", "nervous", "worried", "panic"],
    "confusion": ["confused", "confusing", "puzzled", "dont understand"],
    "curiosity": ["wonder", "curious", "curiosity", "question"],
    "desire": ["want", "wish", "desire", "crave"],
    "disappointment": ["disappointed", "disappointing", "let down"],
    "disapproval": ["disapprove", "disgust", "nah", "nope", "dislike"],
    "disgust": ["disgust", "gross", "nasty", "yuck"],
    "embarrassment": ["embarrass", "embarrassed", "awkward", "cringe"],
    "excitement": ["excited", "cant wait", "thrilled", "pumped"],
    "fear": ["fear", "scared", "afraid", "terrified", "panic"],
    "gratitude": ["thanks", "thank you", "grateful", "appreciate"],
    "grief": ["grief", "grieving", "mourning", "loss"],
    "joy": ["happy", "joy", "delighted", "cheerful", "blessed"],
    "love": ["love", "luv", "adore", "beloved"],
    "neutral": ["ok", "fine", "neutral", "meh"],
    "optimism": ["hope", "optimistic", "looking forward", "hopeful"],
    "pride": ["proud", "pride", "accomplish"],
    "realization": ["realize", "oh i see", "i see now", "realisation"],
    "relief": ["relief", "relieved", "phew"],
    "remorse": ["sorry", "regret", "remorse", "apologize"],
    "sadness": ["sad", "sorrow", "depressed", "unhappy", "cry"],
    "surprise": ["surprise", "surprised", "wow", "unexpected"],
    "sympathy": ["sorry for you", "sympathy", "feel for you", "condolences"]
}

# compile patterns for faster matching
RULE_PATTERNS = []
for label, kwlist in RULE_KEYWORDS.items():
    # create one regex alt group from keywords, escape tokens
    words = [re.escape(w) for w in kwlist if w.strip()]
    if not words:
        continue
    pat = re.compile(r"\b(" + r"|".join(words) + r")\b", flags=re.I)
    RULE_PATTERNS.append((pat, label))

def apply_rules(text):
    """
    Returns (label, pattern) if a rule matches, else (None, None).
    """
    t = "" if text is None else str(text)
    for pat, label in RULE_PATTERNS:
        if pat.search(t):
            return label, pat.pattern
    return None, None

if __name__ == "__main__":
    # quick local test
    samples = ["i am so happy and delighted", "i am furious at this", "i feel sad and cry"]
    for s in samples:
        print(s, "->", apply_rules(s))
