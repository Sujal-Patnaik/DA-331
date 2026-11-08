"""Text normalisation shared by training and the applications (design document, 7.6).

Order matters: URLs go first so ``://`` is not read as an emoticon, and masks are
applied before emoji conversion. Text is never lowercased (design document, 6.5; project rule 4).
"""
from __future__ import annotations

import re
import unicodedata

import emoji

EMOTICONS = {
    ":)": " smiling face ", ":-)": " smiling face ", ":(": " frowning face ",
    ":-(": " frowning face ", ":D": " grinning face ", ":'(": " crying face ",
    "<3": " red heart ", ":/": " confused face ",
}
# A following letter, digit or slash means this is not an emoticon ("Note:Done", "a:/b").
_EMOTICON_RE = re.compile(
    "(?:" + "|".join(re.escape(k) for k in sorted(EMOTICONS, key=len, reverse=True)) + r")(?![\w/])"
)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s-]{8,}\d")
_ID_RE = re.compile(r"\b[A-Z]{2,4}-?\d{5,}\b")
_HANDLE_RE = re.compile(r"(?<![\w.])@\w+")
_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_EMOJI_NAME_RE = re.compile("\ue000([^\ue001]*)\ue001")


def _emoji_to_text(text: str) -> str:
    """Emoji to CLDR short names, with underscores to spaces only inside the name."""
    marked = emoji.demojize(text, delimiters=("\ue000", "\ue001"))
    return _EMOJI_NAME_RE.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", marked)


def mask_pii(text: str) -> str:
    """Mask URLs, e-mails, phones, ticket/roll ids and handles (design document 7.6, project specification, milestone M2)."""
    text = _URL_RE.sub(" [URL] ", text)
    text = _EMAIL_RE.sub(" <email> ", text)
    text = _PHONE_RE.sub(" <phone> ", text)
    text = _ID_RE.sub(" <id> ", text)
    return _HANDLE_RE.sub(" [USER] ", text)


def clean_text(text: str | None) -> str | None:
    """Full cleaning for one text; None stays None."""
    if text is None:
        return None
    text = mask_pii(text)
    text = _EMOTICON_RE.sub(lambda m: EMOTICONS[m.group(0)], text)
    text = _emoji_to_text(text)
    text = unicodedata.normalize("NFC", text)
    return _SPACE_RE.sub(" ", text).strip()
