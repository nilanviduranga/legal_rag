import re

# ── Harmful-query detection ────────────────────────────────────────────────────

_HARMFUL_PATTERNS = [
    r'\bhide\b.{0,40}\btax\b',
    r'\btax\b.{0,40}\bfraud\b',
    r'\btax\b.{0,40}\bevasion\b',
    r'\bevade\b.{0,40}\btax\b',
    r'\bavoid.{0,20}(getting\s+)?caught\b',
    r'\bnot.{0,15}get\s+(caught|detected)\b',
    r'\bdestroy\b.{0,40}\bevidence\b',
    r'\bdispose\b.{0,40}\bevidence\b',
    r'\btamper\b.{0,40}\bevidence\b',
    r'\bhide\b.{0,40}\bevidence\b',
    r'\bconceal\b.{0,40}\bevidence\b',
    r'\bcover.{0,20}(up|illegal|crime|activities)\b',
    r'\bhide.{0,20}(illegal|crime|activities)\b',
    r'\blaunder\b',
    r'\bmoney\s+laundering\b',
    r'\bfraud\b.{0,25}\bscheme\b',
    r'\bscheme\b.{0,25}\bfraud\b',
    r'\bbribery\b',
    r'\bpay.{0,15}\bbribe\b',
    r'\bcorrupt\b.{0,25}\bofficer\b',
    r'\bembezzl\b',
    r'\bget\s+away\s+with\b',
    r'\bcommit\b.{0,25}\b(crime|fraud|offence)\b',
]

_HARMFUL_REFUSAL = """\
I'm sorry, but I'm unable to assist with that request. It appears to involve \
facilitating or concealing illegal activity, which falls outside the scope of \
ethical and lawful legal guidance.

If you have genuine legal concerns, I encourage you to:
• Consult a qualified attorney for confidential, professional advice.
• Contact the relevant regulatory authority for guidance on compliance.
• Review the applicable laws to understand your rights and responsibilities.\
"""


def is_harmful_query(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _HARMFUL_PATTERNS)


def get_refusal_response() -> str:
    return _HARMFUL_REFUSAL


# ── History-relevance detection ────────────────────────────────────────────────

_HISTORY_REF_PATTERNS = [
    r'\bthat\s+act\b',
    r'\bthose\s+acts\b',
    r'\bthe\s+(same|above|mentioned|said)\s+(act|section|provision|law|case|article)\b',
    r'\bthat\s+(section|provision|case|clause|article)\b',
    r'\bthe\s+above\b',
    r'\byou\s+(mentioned|said|explained|noted|referenced|listed)\b',
    r'\bas\s+(you\s+)?(mentioned|explained|stated|noted)\b',
    r'\bwhat\s+you\s+(explained|said|described)\b',
    r'\bcontinue\b',
    r'\bexplain\s+more\b',
    r'\btell\s+me\s+more\b',
    r'\belaborate\b',
    r'\bfurther\s+(explain|elaborate|detail|clarify)\b',
    r'\bmore\s+(detail|info|information|context)\b',
    r'\bcompare\s+(with|to)\b',
    r'\bin\s+(the\s+)?previous\b',
    r'\bfrom\s+(the\s+|your\s+)?(last|previous|above|earlier)\b',
    r'\bearlier\s+(question|message|response|answer|context)\b',
    r'\b(previous|last)\s+(question|message|response|answer)\b',
]


def needs_history(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _HISTORY_REF_PATTERNS)
