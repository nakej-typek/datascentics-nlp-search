"""Text -> tokens for Czech news.

Deliberately simple for the first working version: lowercase, split on word
characters (Python's Unicode ``\\w`` already matches Czech diacritics like
a/c/r -> á/č/ř), then drop a small set of stopwords and pure numbers.

Lemmatization and a diacritics policy are the next quality levers.
We start simple and *measure* before adding complexity — the brief rewards
understanding over piling on libraries.
"""

import re

# Starter Czech stopword list: very common function words that carry little
# topical meaning, so they mostly add noise to retrieval. A fuller list (and
# lemmatization, which would also fold inflected forms together) is a known
# improvement we can justify with evaluation numbers later.
STOPWORDS: set[str] = {
    "a", "aby", "ale", "ani", "ano", "až", "bez", "by", "byl", "byla", "byli",
    "bylo", "být", "co", "či", "do", "ho", "i", "jak", "jako", "je", "jeho",
    "její", "jen", "ještě", "již", "k", "kde", "když", "ke", "která", "které",
    "kteří", "kterou", "který", "má", "máte", "mezi", "mi", "mít", "na", "nad",
    "nám", "námi", "ne", "než", "nic", "o", "od", "po", "pod", "pro", "proč",
    "před", "při", "s", "se", "si", "tak", "také", "te", "to", "tom", "tu",
    "ty", "u", "v", "vám", "ve", "více", "však", "z", "za", "ze", "že",
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into word tokens, drop stopwords and pure numbers."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]
