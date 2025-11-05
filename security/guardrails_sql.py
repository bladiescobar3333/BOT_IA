# security/guardrails_sql.py
from typing import Tuple
from common.sql_utils import (
    sanitize_llm_sql, validate_sql_only_select, reject_select_star, ensure_aliases
)

def secure_sql_pipeline(raw_sql: str) -> Tuple[bool, str, str]:
    """
    Devuelve: (ok, motivo, sql_segura)
    1) Sanitiza bloque triple, 'SQL:', ';'
    2) Valida solo SELECT (sin DDL/DML)
    3) Rechaza SELECT *
    4) Asegura alias
    """
    s = sanitize_llm_sql(raw_sql)
    ok, why = validate_sql_only_select(s)
    if not ok:
        return False, why, s
    ok2, why2 = reject_select_star(s)
    if not ok2:
        return False, why2, s
    s = ensure_aliases(s)
    return True, "OK", s
