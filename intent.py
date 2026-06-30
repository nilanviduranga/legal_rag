from llm import llm_call

_INTENT_LABELS = {"explain", "definition", "compare", "count", "list", "procedure", "penalty", "structural"}

_DISCOVERY_INTENTS = {"count", "list"}

_STRUCTURAL_INTENTS = {"structural"}

_SYSTEM = (
    "Classify the legal question into exactly one label.\n\n"
    "Labels:\n"
    "  structural – asking about the INTERNAL STRUCTURE of a SPECIFIC named Act: how many sections/clauses/provisions it has, or list all sections of a named Act\n"
    "  count      – asking how many laws/acts exist on a topic (NOT about a specific act's structure)\n"
    "  list       – asking to name or enumerate laws/acts that relate to a topic\n"
    "  explain    – asking what a provision or law means\n"
    "  definition – asking what a legal term means\n"
    "  compare    – asking to compare two provisions or acts\n"
    "  procedure  – asking the steps to do something legally\n"
    "  penalty    – asking about punishments, fines, or sentences\n\n"
    "IMPORTANT: Use 'structural' when the question names a specific Act AND asks about sections/clauses/provisions/subsections/parts/chapters within that Act.\n"
    "Use 'count' or 'list' when asking about how many laws COVER A TOPIC (e.g. 'how many laws relate to consumer rights?').\n\n"
    "Examples:\n"
    "  'How many sections are in the Consumer Affairs Authority Act?' → structural\n"
    "  'List all sections of the Labour Act' → structural\n"
    "  'How many provisions does the Penal Code have?' → structural\n"
    "  'How many laws are there about consumer protection?' → count\n"
    "  'List all acts related to employment' → list\n\n"
    "Reply with ONLY the label word — nothing else."
)


def classify_intent(question: str) -> str:
    try:
        label = llm_call(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=10,
        ).strip().lower()
        return label if label in _INTENT_LABELS else "explain"
    except Exception as e:
        print(f"[intent] classification failed: {e}")
        return "explain"


def is_discovery_intent(intent: str) -> bool:
    return intent in _DISCOVERY_INTENTS


def is_structural_intent(intent: str) -> bool:
    return intent in _STRUCTURAL_INTENTS
