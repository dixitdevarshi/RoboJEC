import re
from typing import List

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


# ── NLTK data (downloaded once) ────────────────────────────────────────────────

def _ensure_nltk() -> None:
    for resource, path in [
        ("punkt",                    "tokenizers/punkt"),
        ("averaged_perceptron_tagger","taggers/averaged_perceptron_tagger"),
        ("wordnet",                  "corpora/wordnet"),
        ("stopwords",                "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk()


# ── Number extraction ──────────────────────────────────────────────────────────

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}


def extract_number_from_text(text: str) -> float | None:
    """Return the first number found in text, or None."""
    if not text:
        return None

    digit_match = re.search(r"(\d+\.?\d*)", text)
    if digit_match:
        return float(digit_match.group(1))

    text_lower = text.lower()
    for word, value in _NUMBER_WORDS.items():
        if word in text_lower:
            return float(value)

    return None


# ── Keyword / theme extraction ─────────────────────────────────────────────────

_KEYWORD_SEEDS = [
    "team", "lead", "pressure", "conflict", "challenge", "problem",
    "solution", "manage", "achieve", "result", "learn", "change",
]

_THEME_MAP = {
    "teamwork":    ["team", "collaborat", "work with"],
    "leadership":  ["lead", "manage", "direct"],
    "stress":      ["stress", "pressure", "tense"],
    "conflict":    ["conflict", "disagree", "argument"],
    "achievement": ["achieve", "success", "result"],
    "learning":    ["learn", "grow", "develop"],
}


def extract_keywords(text: str) -> List[str]:
    """Return up to 5 meaningful keywords from text."""
    if not text:
        return []

    tokens   = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)

    found  = [w for w in _KEYWORD_SEEDS if w in text.lower()]
    found += [w for w, p in pos_tags if p.startswith("NN") and len(w) > 3]
    found += [w for w, p in pos_tags if p.startswith("VB") and len(w) > 3]

    return list(set(found))[:5]


def identify_themes(text: str) -> List[str]:
    """Return a list of broad conversation themes detected in text."""
    if not text:
        return []

    text_lower = text.lower()
    themes = [
        theme
        for theme, keywords in _THEME_MAP.items()
        if any(kw in text_lower for kw in keywords)
    ]
    return themes if themes else ["experience"]


def check_star(text: str) -> List[str]:
    """Return the STAR components that are missing from text."""
    missing = []

    if not re.search(r"(when|while|during|situation|circumstance)", text, re.I):
        missing.append("Situation")
    if not re.search(r"(task|goal|objective|needed to|had to)", text, re.I):
        missing.append("Task")
    if not re.search(r"(did|action|step|implement|decided)", text, re.I):
        missing.append("Action")
    if not re.search(r"(result|outcome|achieved|accomplished|learned)", text, re.I):
        missing.append("Result")

    return missing


# ── Hobby extraction ───────────────────────────────────────────────────────────

_STANDALONE_HOBBIES = {
    "dancing", "singing", "painting", "drawing", "reading", "writing",
    "cooking", "baking", "hiking", "swimming", "running", "cycling",
    "gaming", "gardening", "programming", "coding", "photography",
    "traveling", "collecting",
}

_OBJECT_REQUIRING_VERBS = {"play", "playing", "watch", "watching"}

_HOBBY_PHRASES = {
    "watching tv", "watching movies", "playing games", "playing video games",
    "playing music", "listening to music", "working out", "martial arts",
    "playing chess", "playing guitar", "playing piano", "watching sports",
    "playing football", "playing cricket", "watching football", "watching cricket",
}

_HOBBY_INDICATORS = [
    "enjoy", "like", "love", "passion", "hobby", "interest", "into", "fan of",
]


def extract_hobbies(response_text: str) -> List[str]:
    """Extract hobby phrases from a free-text response."""
    if not response_text:
        return []

    text   = response_text.lower()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    potential: list = []

    # indicator-led phrases
    for i, (word, _) in enumerate(tagged):
        if word in _HOBBY_INDICATORS and i < len(tagged) - 1:
            j = i + 1
            phrase_words = []
            verb_found   = None

            while j < len(tagged) and tagged[j][1] not in [".", ",", "CC"] and j < i + 6:
                cw = tagged[j][0]
                phrase_words.append(cw)
                if cw in _OBJECT_REQUIRING_VERBS:
                    verb_found = cw
                j += 1

            if phrase_words:
                phrase = " ".join(phrase_words)
                if verb_found and len(phrase_words) > 1:
                    potential.append(phrase)
                elif not any(w in _OBJECT_REQUIRING_VERBS for w in phrase_words):
                    potential.append(phrase)

    # standalone hobbies
    for word, _ in tagged:
        if word in _STANDALONE_HOBBIES:
            potential.append(word)

    # verb + object combos
    for i, (word, _) in enumerate(tagged):
        if word in _OBJECT_REQUIRING_VERBS and i < len(tagged) - 1:
            j = i + 1
            while j < len(tagged) and tagged[j][1] in ["DT", "JJ", "RB"]:
                j += 1
            objects = []
            while j < len(tagged) and tagged[j][1] in ["NN", "NNS", "NNP", "NNPS"] and j < i + 4:
                objects.append(tagged[j][0])
                j += 1
            if objects:
                potential.append(word + " " + " ".join(objects))

    # n-gram phrase matching
    for n in range(2, 5):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if phrase in _HOBBY_PHRASES:
                potential.append(phrase)

    # deduplicate and normalise
    unique: set = set()
    for hobby in potential:
        if hobby in _OBJECT_REQUIRING_VERBS:
            continue
        hobby = (
            hobby.replace("watch tv", "watching TV")
                 .replace("watching television", "watching TV")
                 .replace("play video games", "playing video games")
                 .replace("play games", "playing games")
        )
        words = hobby.split()
        if (
            (len(words) == 1 and words[0] not in _OBJECT_REQUIRING_VERBS)
            or (words[0] in _OBJECT_REQUIRING_VERBS and len(words) > 1)
            or (words[0] not in _OBJECT_REQUIRING_VERBS)
        ):
            unique.add(hobby)

    return list(unique)