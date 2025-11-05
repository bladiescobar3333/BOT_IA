# security/guardrails_input.py
import re
from typing import Tuple

FORBIDDEN_INPUT_PATTERNS = [
    r"ignore (all|previous) instructions",
    r"override (the|these) rules",
    r"DROP\s+TABLE",
    r"EXEC(\s+|UTE)\b",
    r"CREATE\s+LOGIN",
    r"union\s+select",
]

def sanitize_user_input(msg: str, max_len: int = 2000) -> Tuple[bool, str]:
    s = (msg or "").strip()
    if len(s) > max_len:
        s = s[:max_len]
    for pat in FORBIDDEN_INPUT_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return False, "El mensaje contiene patrones no permitidos."
    return True, s
