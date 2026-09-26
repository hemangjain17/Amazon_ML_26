import re
import unicodedata
from typing import Tuple, Optional

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
except ImportError:
    sanscript = None

# Common legal suffix mappings
LEGAL_SUFFIXES = {
    r"\bprivate limited\b": "pvt ltd",
    r"\bpvt\.?\s*ltd\.?\b": "pvt ltd",
    r"\blimited\b": "ltd",
    r"\bltd\.?\b": "ltd",
    r"\bcorporation\b": "corp",
    r"\binc\.?\b": "inc",
    r"\bincorporated\b": "inc",
    r"\bllc\.?\b": "llc",
    r"\bllp\.?\b": "llp",
    r"\bco\.?\b": "company",
    r"\bcompany\b": "company",
}

# Common address term expansions
ADDRESS_ABBREVIATIONS = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bblvd\.?\b": "boulevard",
    r"\bste\.?\b": "suite",
    r"\bapt\.?\b": "apartment",
    r"\bflr?\.?\b": "floor",
    r"\b2nd\b": "second",
    r"\b1st\b": "first",
    r"\b3rd\b": "third",
}


def transliterate_text(text: str) -> str:
    """Converts non-Latin scripts (Devanagari, Tamil, Accents) to clean ASCII/Latin text."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Fast path: NFD, Mn-stripping and unidecode are all identities on ASCII input.
    if text.isascii():
        return text

    # Unicode Normalization (NFD -> ASCII)
    normalized = unicodedata.normalize("NFD", text)
    ascii_text = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    
    # Unidecode for non-Latin characters
    if unidecode is not None:
        ascii_text = unidecode(ascii_text)
        
    return ascii_text


def _compile_guarded(mapping):
    """Pre-compiles patterns with a literal guard; a pattern cannot match unless its guard substring is present."""
    guards = {
        r"\bprivate limited\b": "private limited",
        r"\bpvt\.?\s*ltd\.?\b": "pvt",
        r"\bflr?\.?\b": "fl",
    }
    compiled = []
    for pattern, replacement in mapping.items():
        guard = guards.get(pattern) or re.match(r"\\b(\w+)", pattern).group(1)
        compiled.append((guard, re.compile(pattern), replacement))
    return compiled


_LEGAL_SUFFIX_RULES = _compile_guarded(LEGAL_SUFFIXES)
_ADDRESS_RULES = _compile_guarded(ADDRESS_ABBREVIATIONS)
_NON_WORD_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Standardizes text string by lowercasing, transliterating, and expanding terms."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Transliterate to ASCII
    text = transliterate_text(text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Standardize Legal Suffixes
    for guard, pattern, replacement in _LEGAL_SUFFIX_RULES:
        if guard in text:
            text = pattern.sub(replacement, text)

    # 4. Standardize Address Terms
    for guard, pattern, replacement in _ADDRESS_RULES:
        if guard in text:
            text = pattern.sub(replacement, text)

    # 5. Strip special characters except alphanumeric and whitespace
    text = _NON_WORD_RE.sub(" ", text)

    # 6. Normalize whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    
    return text


def extract_pin_code(address_str: str) -> Optional[str]:
    """Extracts 5-digit US Zip or 6-digit Indian PIN code if present."""
    if not isinstance(address_str, str):
        return None
    # 6-digit PIN (India) or 5-digit Zip (US)
    match = re.search(r"\b\d{5,6}\b", address_str)
    return match.group(0) if match else None


def format_entity_string(name: str, address: str, country: str = "") -> str:
    """Formats entity record fields into a canonical sequence for transformer embedding."""
    clean_n = clean_text(str(name) if name and str(name).lower() != "nan" else "")
    clean_a = clean_text(str(address) if address and str(address).lower() != "nan" else "")
    c_code = str(country).strip().upper() if country else ""
    
    if c_code:
        return f"[{c_code}] {clean_n} | {clean_a}"
    return f"{clean_n} | {clean_a}"
