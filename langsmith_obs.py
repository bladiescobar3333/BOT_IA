# observability/langsmith_obs.py
import os, time
from typing import Dict, Any

def setup_langsmith(project: str = "POI-Prod"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    # LANGCHAIN_API_KEY debe estar en el entorno
    return True

def audit_event(event_type: str, payload: Dict[str, Any]):
    ts = int(time.time())
    print(f"[AUDIT][{ts}][{event_type}] {payload}")
