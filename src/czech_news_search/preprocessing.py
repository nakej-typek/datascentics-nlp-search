"""Text -> tokens for Czech news.

Deliberately simple for the first working version: lowercase, split on word
characters (Python's Unicode ``\\w`` already matches Czech diacritics like
a/c/r -> á/č/ř), then drop a small set of stopwords and pure numbers.

Lemmatization and a diacritics policy are the next quality levers.
We start simple and *measure* before adding complexity — the brief rewards
understanding over piling on libraries.
"""

import re
from functools import lru_cache

import simplemma

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


@lru_cache(maxsize=100_000)
def _lemma(word: str) -> str:
    """Czech lemma of a single word (cached -- the same words recur a lot).

    Folds inflected forms together: povodně/povodní -> povodeň, vlády -> vláda.
    This is the biggest single-lever recall win for a heavily inflected language
    like Czech: a query in one form now matches documents using another form.
    """
    return simplemma.lemmatize(word, lang="cs")


def tokenize(text: str, lemmatize: bool = True) -> list[str]:
    """Lowercase, split into word tokens, drop stopwords/numbers, optionally lemmatize.

    ``lemmatize`` defaults to True (the production pipeline); pass False to get
    the raw-token baseline for before/after comparison in evaluation.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and not t.isdigit()]
    if lemmatize:
        # Lemmatize first, then drop any stopword the lemma collapsed onto.
        tokens = [lemma for t in tokens if (lemma := _lemma(t)) not in STOPWORDS]
    return tokens
