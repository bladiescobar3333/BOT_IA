# mcp/server.py
import os, sys, json, pyodbc, datetime, decimal
from typing import Any, Dict
from dotenv import load_dotenv

# Carga variables del entorno (.env)
load_dotenv()  # carga MCP_PRD_DSN, MCP_CFA_DSN, etc.

# ---------- JSON helpers ----------
def _json_default(o):
    if isinstance(o, (datetime.date, datetime.datetime, datetime.time)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
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

def _ok(payload: Dict[str, Any] | None = None):
    data = {"ok": True}
    if payload:
        data.update(payload)
    _json_print(data)

def _fail(msg: str, extra: Dict[str, Any] | None = None):
    data = {"ok": False, "error": msg}
    if extra:
        data.update(extra)
    _json_print(data)

# ---------- DSN / ODBC ----------
def _dsn_from_env(alias: str) -> str:
    a = (alias or "").strip().lower()
    if a == "prd":  return os.getenv("MCP_PRD_DSN", "")
    if a == "qa":   return os.getenv("MCP_QA_DSN", "")
    if a in ("his", "hist"): return os.getenv("MCP_HIS_DSN", "")
    if a in ("cfa", "repl"): return os.getenv("MCP_CFA_DSN", "")
    return ""

def _open_conn(dsn: str) -> pyodbc.Connection:
    cn = pyodbc.connect(dsn, autocommit=False)
    qto = int(os.getenv("MCP_QUERY_TIMEOUT", "30"))
    try:
        cn.timeout = qto
    except Exception:
        pass
    return cn

# ---------- SharePoint (Device Code) ----------
try:
    from .sp_client_device import read_excel_from_sharepoint_device
    _HAS_SP_DEVICE = True
except Exception:
    _HAS_SP_DEVICE = False

def _sp_read_device(site: str, path: str):
    if not _HAS_SP_DEVICE:
        raise RuntimeError("No se encontró sp_client_device.py o falló el import.")
    cols, rows = read_excel_from_sharepoint_device(site, path)
    return cols, rows

# ---------- Main ----------
def main():
    # Lee el request JSON desde stdin (soporta interactivo y pipes)
    try:
        if sys.stdin.isatty():
            # Si está interactivo (por teclado), lee una línea
            raw = sys.stdin.readline().strip()
        else:
            # Si viene por pipe o subprocess, lee todo el stdin
            raw = sys.stdin.read().strip()

        if not raw:
            _fail("Entrada vacía en stdin (envía un JSON válido o usa pipe).")
            return

        req = json.loads(raw)
    except Exception as e:
        _fail(f"Entrada no es JSON: {e}")
        return

    tool = (req.get("tool") or "").lower()
    db   = (req.get("db") or "").lower()

    # ---- Tool: ping ----
    if tool == "sql.ping":
        dsn = _dsn_from_env(db or "prd")
        if not dsn:
            _fail(f"DB desconocida o sin DSN configurado: {db}")
            return
        try:
            cn = _open_conn(dsn)
            try:
                _ok({"message": "Conexión OK", "dsn_used": dsn})
            finally:
                try: cn.close()
                except Exception: pass
        except Exception as e:
            _fail(str(e))
        return

    # ---- Tool: readonly ----
    if tool == "sql.readonly":
        dsn = _dsn_from_env(db)
        if not dsn:
            _fail(f"DB desconocida o sin DSN configurado: {db}")
            return

        query = (req.get("query") or "").strip()
        if not query.lower().startswith("select"):
            _fail("Solo SELECT permitido")
            return

        cn = None
        cur = None
        try:
            cn = _open_conn(dsn)
            cur = cn.cursor()
            cur.execute(query)

            cols = [c[0] for c in (cur.description or [])]
            rows = []
            for row in cur.fetchall():
                d = {}
                for k, v in zip(cols, row):
                    d[k] = _json_default(v) if not isinstance(v, (str, int, float, type(None))) else v
                rows.append(d)

            _ok({"columns": cols, "rows": rows})
        except pyodbc.Error as e:
            msg = getattr(e, "args", [str(e)])
            _fail("Error ODBC al ejecutar SELECT", {"details": msg})
        except Exception as e:
            _fail(str(e))
        finally:
            try:
                if cur is not None:
                    cur.close()
            except Exception:
                pass
            try:
                if cn is not None:
                    cn.close()
            except Exception:
                pass
        return

    # ---- Tool: SharePoint (Device Code) ----
    if tool == "sp.file.device":
        site = req.get("site") or os.getenv("SP_DEFAULT_SITE", "")
        path = req.get("path")
        if not site:
            _fail("Falta 'site' (URL del sitio de SharePoint)")
            return
        if not path:
            _fail("Falta 'path' (ruta server-relative del archivo en SharePoint)")
            return
        try:
            cols, rows = _sp_read_device(site, path)
            _ok({"columns": cols, "rows": rows})
        except Exception as e:
            _fail(str(e))
        return

    # ---- Tool desconocida ----
    _fail(f"Tool no soportada: {tool}")

if __name__ == "__main__":
    main()
