# mcp/db_registry.py
import os, pyodbc
from typing import Optional

ALIASES = {
    "prd": "MCP_PRD_DSN",
    "qa":  "MCP_QA_DSN",
    "his": "MCP_HIS_DSN",
    "cfa": "MCP_CFA_DSN",      # <<< nuevo
    "repl":"MCP_CFA_DSN",      # <<< opcional (sinónimo)
}


def get_dsn(alias: str) -> Optional[str]:
    if not alias:
        return None
    env_key = ALIASES.get(alias.lower())
    if not env_key:
        return None
    return os.getenv(env_key)

def open_conn(dsn: str):
    # dsn debe ser un connection string completo ODBC/SQL Server
    # Ejemplo en .env:
    # MCP_PRD_DSN=Driver={ODBC Driver 17 for SQL Server};Server=tcp:mi-servidor.database.windows.net,1433;Database=db_bi_production_prd;Uid=user;Pwd=pass;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30
    return pyodbc.connect(dsn, timeout=30)
