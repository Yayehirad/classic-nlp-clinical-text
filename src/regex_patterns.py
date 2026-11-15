"""Regular-expression tools: a rule-based chatbot and clinical information extraction."""

import re

# ---------------------------------------------------------------------------
# Part A: CodePal, a rule-based (Eliza-style) helper for people learning to code
# ---------------------------------------------------------------------------

CODEPAL_RULES = [
    (r"\b(bug|bugs|debug|debugging)\b",
     "Debugging can be frustrating. Have you tried printing intermediate results?"),
    (r"\b(won't compile|compile|error|errors|crashing|crashes|crash)\b",
     "Let's take a deep breath and go step-by-step. What does the error message say?"),
    (r"\b(don't understand|dont understand|confusing|confused|too hard|hard to understand)\b",
     "That's a common hurdle. Would you like a resource that explains it visually?"),
    (r"\b(who are you|what are you|are you human|are you a bot)\b",
     "I'm CodePal, your friendly code companion designed to support your learning journey."),
]
FAREWELL_BASIC = r"\bbye\b"
FAREWELL_EXTENDED = r"\b(bye|gotta go|exit|logout|see you)\b"
FAREWELL_REPLY = "Happy coding! Remember, every bug is a step forward."
FALLBACK_REPLY = "Let's tackle it together — what's your next coding challenge?"


def codepal_response(message, extended=False):
    """Return CodePal's reply to one message.

    Rules are checked in order and the first match wins. With extended=True,
    "gotta go", "exit", "logout" and "see you" also end the chat.
    """
    msg = message.lower()
    for pattern, reply in CODEPAL_RULES:
        if re.search(pattern, msg):
            return reply
    if re.search(FAREWELL_EXTENDED if extended else FAREWELL_BASIC, msg):
        return FAREWELL_REPLY
    return FALLBACK_REPLY


def is_farewell(message, extended=False):
    """True if the message should end the conversation."""
    return bool(re.search(FAREWELL_EXTENDED if extended else FAREWELL_BASIC, message.lower()))


def chat(extended=False):
    """Run an interactive chat in the terminal until the user says goodbye."""
    print("Start talking to CodePal. Say 'bye' to finish.\n")
    while True:
        user_input = input("You: ").strip()
        print(f"CodePal: {codepal_response(user_input, extended)}\n")
        if is_farewell(user_input, extended):
            break


def simulate_chat(messages, extended=False):
    """Play a scripted conversation (non-interactive version of chat())."""
    for message in messages:
        print(f"You: {message}")
        print(f"CodePal: {codepal_response(message, extended)}\n")
        if is_farewell(message, extended):
            break


# ---------------------------------------------------------------------------
# Part B: extracting doses, vital signs and dates from clinical notes
# ---------------------------------------------------------------------------

_NUM = r"\d+(?:\.\d+)?"
_UNIT = r"(?:mcg|mg|g|mL|ml|units?|IU|mEq|mmol)"

CLINICAL_PATTERNS = {
    # 500 mg, 0.5 mg, 10 mg/kg, 2 g, 40 units, 5 mL, 1-2 mg
    "dose": re.compile(
        rf"\b{_NUM}(?:\s?-\s?{_NUM})?\s?{_UNIT}(?:\s?/\s?(?:kg|day|hr|hour|h|dose|m2))?(?![A-Za-z])",
        re.IGNORECASE),
    # blood pressure 120/80, BP: 130/85 mmHg
    "blood_pressure": re.compile(
        r"\b(?:blood pressure|BP)\b[^0-9]{0,25}?(\d{2,3}\s?/\s?\d{2,3})(?:\s?mm\s?Hg)?",
        re.IGNORECASE),
    # heart rate 72, pulse 88, HR: 110
    "heart_rate": re.compile(
        r"\b(?:heart rate|pulse(?: rate)?|HR)\b[^0-9]{0,20}?(\d{2,3})\b",
        re.IGNORECASE),
    # temperature 98.6, temp 37.2 C, T: 101.2
    "temperature": re.compile(
        r"\b(?:temperature|temp|T(?=\s?[:=]))\b[^0-9]{0,20}?(\d{2,3}(?:\.\d)?)\s?(?:degrees\s?)?(?:°\s?)?([FC]\b)?",
        re.IGNORECASE),
    # 01/15/2004, 1-15-04, January 15, 2004, Jan 2004
    "date": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})"
        r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
        r"|(?:January|February|March|April|June|July|August|September|October|November|December)\s+\d{4})\b"),
}


def extract_clinical_entities(text, context=40):
    """Find every dose, vital sign and date in a note.

    Returns a list of dicts with the entity type, the matched text and a short
    window of surrounding text (useful for checking matches by hand).
    """
    found = []
    for entity, pattern in CLINICAL_PATTERNS.items():
        for m in pattern.finditer(text):
            start, end = m.span()
            found.append({
                "entity": entity,
                "match": m.group(0),
                "context": text[max(0, start - context):min(len(text), end + context)].replace("\n", " "),
                "start": start,
            })
    return found
