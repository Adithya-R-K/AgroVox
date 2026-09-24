"""
AgroVox - Text preprocessing module.

Implements the Phase 1 preprocessing pipeline shared by every downstream
NLP module: normalization, cleaning, tokenization, stopword removal and
lemmatization for English, plus light Tamil-specific normalization
(Tamil has no meaningful English-style stemming/stopword library available
offline, so Tamil text is normalized and tokenized but not lemmatized -
this limitation is documented here and in the README).
"""
import re
import string
import unicodedata

import nltk

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize

try:
    _EN_STOPWORDS = set(nltk_stopwords.words("english"))
except LookupError:
    # minimal offline fallback stopword list if the NLTK corpus download failed
    _EN_STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "am",
        "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
        "her", "its", "our", "their", "of", "to", "in", "on", "at", "for",
        "with", "and", "or", "but", "if", "so", "this", "that", "these",
        "those", "do", "does", "did", "have", "has", "had", "not", "no",
    }

_LEMMATIZER = WordNetLemmatizer()
_STEMMER = SnowballStemmer("english")

# Words that carry real agronomic meaning even though they look like stopwords
# in general English (e.g. "will" in "will it rain") - keep these.
_KEEP_WORDS = {"not", "no"}
_EN_STOPWORDS = _EN_STOPWORDS - _KEEP_WORDS

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_unicode(text: str) -> str:
    """NFC-normalize unicode so Tamil (and English) glyphs compare consistently."""
    return unicodedata.normalize("NFC", text or "")


def clean_text(text: str, lowercase: bool = True) -> str:
    """Strip URLs/extra whitespace, normalize unicode, optionally lowercase.
    Punctuation is intentionally kept out of this step (handled at tokenization)
    so Tamil combining marks are not damaged by aggressive regex punctuation strips.
    """
    text = normalize_unicode(text)
    text = _URL_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def is_tamil(text: str) -> bool:
    """Heuristic language check: does the text contain Tamil Unicode block characters?"""
    return any("\u0B80" <= ch <= "\u0BFF" for ch in (text or ""))


def detect_language(text: str) -> str:
    """Very lightweight language identification: Tamil-script vs English.
    Returns an ISO-ish code: 'ta' or 'en'. This is sufficient for AgroVox's
    two supported languages; a general-purpose langid model would be used
    for a broader multilingual system.
    """
    text = text or ""
    tamil_chars = sum(1 for ch in text if "\u0B80" <= ch <= "\u0BFF")
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return "en"
    return "ta" if tamil_chars / max(letters, 1) > 0.3 else "en"


def tokenize_english(text: str):
    text = clean_text(text)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.translate(_PUNCT_TABLE).split()
    return [t for t in tokens if t.strip(string.punctuation)]


def tokenize_tamil(text: str):
    """Tamil tokenization: whitespace + punctuation split (no dedicated Tamil
    tokenizer is bundled offline). Preserves Tamil grapheme clusters intact
    since we never break mid-character."""
    text = normalize_unicode(text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    tokens = re.findall(r"[\u0B80-\u0BFF]+|[A-Za-z0-9]+", text)
    return tokens


def remove_stopwords_en(tokens):
    return [t for t in tokens if t.lower() not in _EN_STOPWORDS]


def lemmatize_en(tokens):
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def stem_en(tokens):
    """Stemming (in addition to lemmatization) collapses inflectional
    variants like 'yellow'/'yellowing' or 'irrigate'/'irrigation' to a
    shared root, which is valuable recall signal for bag-of-words TF-IDF
    retrieval (QA engine) even though it is too aggressive for display."""
    return [_STEMMER.stem(t) for t in tokens]


def preprocess(text: str, language: str = None, remove_stopwords: bool = True,
               lemmatize: bool = True) -> dict:
    """Full preprocessing pipeline entry point.

    Returns a dict with the cleaned text, detected/declared language,
    tokens, and (for English) stopword-removed + lemmatized tokens, ready
    to feed downstream classification/NER/QA modules.
    """
    raw = text or ""
    lang = language or detect_language(raw)

    if lang == "ta":
        cleaned = clean_text(raw, lowercase=False)  # Tamil script has no case
        tokens = tokenize_tamil(cleaned)
        processed_tokens = tokens  # no stopword/lemmatization resource available offline
    else:
        cleaned = clean_text(raw, lowercase=True)
        tokens = tokenize_english(cleaned)
        processed_tokens = tokens
        if remove_stopwords:
            processed_tokens = remove_stopwords_en(processed_tokens)
        if lemmatize:
            processed_tokens = lemmatize_en(processed_tokens)

    stemmed_tokens = stem_en(processed_tokens) if lang == "en" else processed_tokens

    return {
        "raw_text": raw,
        "cleaned_text": cleaned,
        "language": lang,
        "tokens": tokens,
        "processed_tokens": processed_tokens,
        "processed_text": " ".join(processed_tokens),
        "stemmed_tokens": stemmed_tokens,
        "stemmed_text": " ".join(stemmed_tokens),
    }


if __name__ == "__main__":
    samples = [
        "My rice crop has yellow leaves!! What should I do??",
        "என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது. என்ன செய்ய வேண்டும்?",
    ]
    for s in samples:
        print(preprocess(s))
