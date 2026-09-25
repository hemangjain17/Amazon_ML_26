import re
import unicodedata
from typing import Optional

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None

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
    if not isinstance(text, str) or not text.strip():
        return ""
    normalized = unicodedata.normalize("NFD", text)
    ascii_text = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    if unidecode is not None:
        ascii_text = unidecode(ascii_text)
    return ascii_text


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = transliterate_text(text)
    text = text.lower()
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_pin_code(address_str: str) -> Optional[str]:
    if not isinstance(address_str, str):
        return None
    match = re.search(r"\b\d{5,6}\b", address_str)
    return match.group(0) if match else None


def format_entity_string(name: str, address: str, country: str = "") -> str:
    clean_n = clean_text(str(name) if name and str(name).lower() != "nan" else "")
    clean_a = clean_text(str(address) if address and str(address).lower() != "nan" else "")
    c_code = str(country).strip().upper() if country else ""
    if c_code:
        return f"[{c_code}] {clean_n} | {clean_a}"
    return f"{clean_n} | {clean_a}"
