"""Privacy helpers (PII redaction)."""

from __future__ import annotations

import re

# Conservative patterns: target common contact/identity PII.
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
CONTACT_URL_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com)/[^\s)]+\b", re.IGNORECASE
)
ADDRESS_LINE_RE = re.compile(
    r"(?im)^\s*\d{1,6}\s+.*\b(st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|way|pkwy|parkway|pl|place)\b.*$"
)


def redact_pii(text: str) -> str:
    """
    Redact likely PII from text.

    Intended use: reduce accidental leakage of contact info to cloud providers.
    Conservative by design: redacts emails, US-style phone numbers, LinkedIn/GitHub URLs,
    and street-like address lines.
    """
    if not text:
        return text

    out = text
    # Redact whole address lines first.
    out = ADDRESS_LINE_RE.sub("[REDACTED_ADDRESS]", out)
    # Redact common contact details.
    out = EMAIL_RE.sub("[REDACTED_EMAIL]", out)
    out = PHONE_RE.sub("[REDACTED_PHONE]", out)
    out = CONTACT_URL_RE.sub("[REDACTED_URL]", out)
    return out

