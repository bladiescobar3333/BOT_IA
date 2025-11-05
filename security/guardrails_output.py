# security/guardrails_output.py
import os
from typing import List, Dict, Any, Optional

def limit_rows(rows: List[Dict[str, Any]], max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    if max_rows is None:
        max_rows = int(os.getenv("MAX_ROWS", "100000"))
    return rows[:max_rows]

def get_whitelist_from_env() -> Optional[List[str]]:
    """
    Lee OUTPUT_WHITELIST del .env.
    - '*' o vacío => sin filtro (None)
    - Caso contrario => lista de columnas en MAYÚSCULAS
    """
    raw = (os.getenv("OUTPUT_WHITELIST", "*") or "").strip()
    if raw == "" or raw == "*":
        return None
    return [x.strip().upper() for x in raw.split(",") if x.strip()]

def whitelist_columns(rows: List[Dict[str, Any]], whitelist: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Aplica whitelist de forma case-insensitive.
    Si whitelist es None => no filtra.
    """
    if not rows or not whitelist:
        return rows

    wl = set(whitelist)  # ya viene en MAYÚSCULAS
    out: List[Dict[str, Any]] = []
    for r in rows:
        # conserva la clave original pero compara en MAYÚSCULAS
        filtered = {k: v for k, v in r.items() if k.upper() in wl}
        out.append(filtered)
    return out
