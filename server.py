# mcp/server.py
import os
import sys
import json
import pyodbc
import datetime
import decimal
from typing import Any, Dict

# ------------------------------
# Helpers de serialización JSON
# ------------------------------
def _json_default(o):
    if isinstance(o, (datetime.date, datetime.datetime, datetime.time)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
        # usa float; cambia a str(o) si prefieres exactitud decimal
        return float(o)
    if isinstance(o, bytes):
        try:
            return o.decode("utf-8", errors="ignore")
        except Exception:
            return repr(o)
    return str(o)

def _json_print(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=_json_default))
    sys.stdout.flush()

# ------------------------------
# Carga DSN desde variables .env
# ------------------------------
def _dsn_from_env(alias: str) -> str:
    alias = (alias or "").strip().lower()
    if alias == "prd":
        return os.getenv("MCP_PRD_DSN", "")
    if alias == "qa":
        return os.getenv("MCP_QA_DSN", "")
    if alias in ("his", "hist"):
        return os.getenv("MCP_HIS_DSN", "")
    return ""

def _open_conn(dsn: str) -> pyodbc.Connection:
    cn = pyodbc.connect(dsn, autocommit=False)
    # Timeout de consulta en segundos (opcional)
    qto = int(os.getenv("MCP_QUERY_TIMEOUT", "30"))
    try:
        # pyodbc usa timeout en el objeto CONEXIÓN, no en el cursor
        cn.timeout = qto
    except Exception:
        pass
    return cn

# ------------------------------
# Main
# ------------------------------
def main():
    try:
        req = json.loads(sys.stdin.read() or "{}")
    except Exception as e:
        _json_print({"ok": False, "error": f"Entrada no es JSON: {e}"})
        return

    tool = req.get("tool")
    db   = (req.get("db") or "").lower()

    dsn = _dsn_from_env(db)
    if not dsn:
        _json_print({"ok": False, "error": f"DB desconocida o sin DSN configurado: {db}"})
        return

    # Ping de conectividad
    if tool == "sql.ping":
        try:
            cn = _open_conn(dsn)
            cn.close()
            _json_print({"ok": True, "message": "Conexión OK", "dsn_used": dsn})
        except Exception as e:
            _json_print({"ok": False, "error": str(e)})
        return

    # Solo lectura
    if tool == "sql.readonly":
        query = (req.get("query") or "").strip()
        if not query.lower().startswith("select"):
            _json_print({"ok": False, "error": "Solo SELECT permitido"})
            return

        try:
            cn = _open_conn(dsn)
            cur = cn.cursor()

            # IMPORTANTE: el timeout va en la conexión, NO en el cursor
            # cur.timeout = ...  # <- esto causaba el error; no usar

            cur.execute(query)
            cols = [c[0] for c in (cur.description or [])]

            rows = []
            for row in cur.fetchall():
                # convierte cada fila en dict + saneo de tipos
                d = {}
                for k, v in zip(cols, row):
                    if isinstance(v, bytes):
                        try:
                            v = v.decode("utf-8", errors="ignore")
                        except Exception:
                            v = repr(v)
                    elif isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
                        v = v.isoformat()
                    elif isinstance(v, decimal.Decimal):
                        v = float(v)  # o str(v)
                    d[k] = v
                rows.append(d)

            cur.close()
            cn.close()

            _json_print({"ok": True, "columns": cols, "rows": rows})
        except Exception as e:
            _json_print({"ok": False, "error": str(e)})
        return

    _json_print({"ok": False, "error": f"Tool no soportada: {tool}"})


if __name__ == "__main__":
    main()
