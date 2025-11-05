# common/sql_utils.py
import re
from typing import Tuple

# Detecta solo SELECT al inicio (ignora espacios, saltos, comentarios)
READ_ONLY_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE | re.DOTALL)

# Corrige el typo "E  XEC" → "EXEC" y endurece detección DDL/DML/EXEC
DDL_DML_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|ALTER|DROP|TRUNCATE|CREATE|GRANT|EXEC|EXECUTE)\b",
    re.IGNORECASE
)

# Detecta SELECT * aunque haya tabs o saltos de línea
STAR = re.compile(r"select\s*\*\s*from", re.IGNORECASE | re.DOTALL)


def sanitize_llm_sql(s: str) -> str:
    """Limpia SQL generado por el LLM: elimina fences, etiquetas y ';' finales."""
    s = str(s or "").strip()
    s = re.sub(r"^```sql\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^\s*SQL\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r";+\s*$", "", s)
    return s


def validate_sql_only_select(sql: str) -> Tuple[bool, str]:
    """Valida que la consulta sea solo SELECT (sin DDL/DML/EXEC)."""
    if not READ_ONLY_PATTERN.search(sql):
        return False, "La consulta no comienza con SELECT."
    if DDL_DML_PATTERN.search(sql):
        return False, "La consulta contiene operaciones no permitidas (DDL/DML)."
    return True, "OK"


def reject_select_star(sql: str) -> Tuple[bool, str]:
    """Bloquea SELECT * y sugiere columnas explícitas."""
    if STAR.search(sql):
        return False, "Evita 'SELECT *'; usa columnas explícitas."
    return True, "OK"


def ensure_aliases(sql: str) -> str:
    """
    Inyecta alias para expresiones comunes (evita error 8155 al envolver).
    No renombra columnas simples (Z.PLANTA, PD.MAREA, etc.).
    """
    m = re.search(r"^\s*SELECT\s+(.*?)\s+FROM\s", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return sql

    parts = [p.strip() for p in m.group(1).split(",")]
    new_parts = []

    for i, expr in enumerate(parts, 1):
        low = expr.lower()
        # Si ya tiene alias o es una columna simple, dejar igual
        if re.search(r"\bas\s+\w+", low) or re.fullmatch(r"[a-zA-Z_][\w\.]*", expr):
            new_parts.append(expr)
            continue

        # Alias automáticos según función
        if "year(" in low:
            new_parts.append(f"{expr} AS ANIO"); continue
        if "month(" in low:
            new_parts.append(f"{expr} AS MES"); continue
        if "sum(" in low:
            new_parts.append(f"{expr} AS TOTAL"); continue
        if "count(" in low:
            new_parts.append(f"{expr} AS CNT"); continue
        if "avg(" in low:
            new_parts.append(f"{expr} AS PROM"); continue
        if "min(" in low:
            new_parts.append(f"{expr} AS MIN"); continue
        if "max(" in low:
            new_parts.append(f"{expr} AS MAX"); continue

        # Alias genérico si nada anterior aplica
        new_parts.append(f"{expr} AS COL{i}")

    return re.sub(
        r"^\s*SELECT\s+(.*?)\s+FROM\s",
        f"SELECT {', '.join(new_parts)} FROM ",
        sql,
        flags=re.IGNORECASE | re.DOTALL
    )
