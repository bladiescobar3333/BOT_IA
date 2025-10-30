# app_langgraph.py
import os, sys, json, shlex, subprocess, re
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from typing import Literal, Dict, Any, TypedDict, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from common.prompts import SYSTEM_PROMPT, INTENT_TEMPLATE, SQL_TEMPLATE, CHAT_TEMPLATE
from security.guardrails_input import sanitize_user_input
from security.guardrails_sql import secure_sql_pipeline
from security.guardrails_output import limit_rows, whitelist_columns, get_whitelist_from_env
from observability.langsmith_obs import setup_langsmith, audit_event

# =========================
# .env + LangSmith (solo si hay API key)
# =========================
load_dotenv()
if os.getenv("LANGCHAIN_API_KEY"):
    setup_langsmith(project=os.getenv("LANGCHAIN_PROJECT", "POI-Prod"))
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# =========================
# UI
# =========================
st.set_page_config(page_title="Agente de Planificación — LangGraph+MCP", page_icon="🧠", layout="wide")
st.title("🧠 Agente de Planificación (LangGraph + MCP + Guardrails + LangSmith)")

with st.sidebar:
    st.header("🔐 Claves")
    google_key = st.text_input("GOOGLE_API_KEY", type="password", value=os.getenv("GOOGLE_API_KEY", ""))

    st.header("⚙️ Parámetros")
    target_table = st.text_input("Tabla objetivo", value="sap.ZQM_DESC_MP_CHI2")
    show_sql_debug = st.checkbox("Mostrar SQL generada (debug)", value=True)

    st.header("🧩 MCP")
    default_mcp = f"\"{sys.executable}\" -m mcp.server"
    mcp_cmd = st.text_input("MCP cmd (opcional)", value=default_mcp)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Probar MCP", use_container_width=True):
            test_req = {"tool": "sql.ping", "db": "prd"}
            try:
                proc = subprocess.Popen(
                    shlex.split(mcp_cmd),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                out, err = proc.communicate(input=json.dumps(test_req), timeout=45)
                if err:
                    st.warning(f"[MCP][stderr]\n{err[:800]}")
                try:
                    res = json.loads(out or "{}")
                except Exception as je:
                    res = {"ok": False, "error": f"Salida MCP no es JSON válido: {je}", "raw": out}

                if res.get("ok"):
                    msg = res.get("message", "Ping OK")
                    dsn_used = res.get("dsn_used", "")
                    st.success(f"✅ MCP OK: {msg}\n\nDSN usado: {dsn_used}")
                else:
                    st.error(f"❌ MCP fallo: {res.get('error', 'sin detalle')}")
            except Exception as e:
                st.error(f"❌ No se pudo ejecutar MCP: {e}")

    with c2:
        st.caption("Tip: Mantén este comando; ya utiliza tu venv correctamente.")

    st.header("💾 RAG (opcional)")
    use_rag = st.checkbox("Usar RAG (no requerido para empezar)", value=False)

st.caption("Tip: 'suma DESCARG CALLAO 2024 en prd', 'total DECLARA VEGUETA 2023 en qa', 'Qué es RAP'.")

# =========================
# LLM helpers
# =========================
def make_llm(temperature: float = 0.2):
    if not (google_key or "").strip():
        st.warning("Falta GOOGLE_API_KEY.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature, api_key=google_key)

llm_router = make_llm(0.1)
llm_chat   = make_llm(0.2)

intent_prompt = PromptTemplate.from_template(INTENT_TEMPLATE)
sql_prompt    = PromptTemplate.from_template(SQL_TEMPLATE)
chat_prompt   = PromptTemplate.from_template(CHAT_TEMPLATE)

# =========================
# Estado
# =========================
class POIState(TypedDict):
    message: str
    intent: Literal["SQL","CHAT",""]
    sql: str
    db: Literal["prd","qa","his",""]
    rows: List[Dict[str, Any]]
    cols: List[str]
    reply: str
    chat_ctx: str
    rag_ctx: str

# =========================
# Adaptadores SQL Server
# =========================
def normalize_sqlserver(sql: str) -> str:
    """
    - Quita ';' final.
    - LIMIT n           -> SELECT TOP n ...
    - LIMIT n OFFSET m / OFFSET m LIMIT n -> ... ORDER BY FECHA DESC OFFSET m ROWS FETCH NEXT n ROWS ONLY
    """
    s = (sql or "").strip()
    s = re.sub(r";\s*$", "", s)

    m_off_lim = re.search(r"\bOFFSET\s+(\d+)\s+LIMIT\s+(\d+)\b", s, flags=re.IGNORECASE)
    m_lim_off = re.search(r"\bLIMIT\s+(\d+)\s+OFFSET\s+(\d+)\b", s, flags=re.IGNORECASE)

    if m_off_lim or m_lim_off:
        if m_off_lim:
            off = int(m_off_lim.group(1)); lim = int(m_off_lim.group(2))
            s = re.sub(r"\bOFFSET\s+\d+\s+LIMIT\s+\d+\b", "", s, flags=re.IGNORECASE)
        else:
            lim = int(m_lim_off.group(1)); off = int(m_lim_off.group(2))
            s = re.sub(r"\bLIMIT\s+\d+\s+OFFSET\s+\d+\b", "", s, flags=re.IGNORECASE)

        if not re.search(r"\bORDER\s+BY\b", s, flags=re.IGNORECASE):
            s += " ORDER BY FECHA DESC"
        s += f" OFFSET {off} ROWS FETCH NEXT {lim} ROWS ONLY"
        return s

    m_lim = re.search(r"\bLIMIT\s+(\d+)\b\s*$", s, flags=re.IGNORECASE)
    if m_lim:
        n = int(m_lim.group(1))
        s = re.sub(r"\bLIMIT\s+\d+\b\s*$", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*SELECT\s", f"SELECT TOP {n} ", s, flags=re.IGNORECASE)
        return s

    return s

def strip_auto_aliases(sql: str) -> str:
    """Elimina alias genéricos tipo AS COL1, AS COL2, ..."""
    return re.sub(r"(?i)\s+AS\s+COL\d+\b", "", sql or "")

def _maybe_remap_cols(cols: List[str], rows: List[Dict[str, Any]]) -> (List[str], List[Dict[str, Any]]):
    """
    Si por algún motivo siguen llegando COL1..COLn y el usuario definió OUTPUT_WHITELIST,
    remapea usando esos nombres (en orden).
    """
    if not cols:
        return cols, rows
    if all(re.fullmatch(r"(?i)COL\d+", (c or "")) for c in cols):
        wl = get_whitelist_from_env()
        if wl and len(wl) >= len(cols):
            new_cols = wl[:len(cols)]
            new_rows = [{new_cols[i]: r.get(cols[i]) for i in range(len(cols))} for r in rows]
            return new_cols, new_rows
    return cols, rows

# =========================
# Nodos
# =========================
def node_router(state: POIState):
    ok, clean = sanitize_user_input(state["message"])
    if not ok:
        return {**state, "intent": "", "reply": "Tu mensaje contiene patrones no permitidos."}

    raw = (intent_prompt | llm_router | StrOutputParser()).invoke({
        "system": SYSTEM_PROMPT.format(target_table=target_table),
        "message": clean
    })
    lab = (raw or "").strip().upper()
    intent = "SQL" if lab.startswith("SQL") else "CHAT"
    audit_event("router", {"message": clean, "intent": intent})
    return {**state, "message": clean, "intent": intent}

def node_plan_sql(state: POIState):
    raw = (sql_prompt | llm_router | StrOutputParser()).invoke({
        "system": SYSTEM_PROMPT.format(target_table=target_table),
        "target_table": target_table,
        "question": state["message"]
    })
    ok, why, sql = secure_sql_pipeline(raw)
    if ok:
        sql = normalize_sqlserver(sql)
        sql = strip_auto_aliases(sql)   # <<< evita COL1..COLn
    audit_event("plan_sql", {"ok": ok, "why": why, "sql": (sql or "")[:400]})
    if not ok:
        return {**state, "sql": "", "reply": f"SQL inválida: {why}"}
    return {**state, "sql": sql}

def node_pick_db(state: POIState):
    m = state["message"].lower()
    db = "prd"
    if " en qa" in m: db = "qa"
    if " en his" in m or " en hist" in m: db = "his"
    audit_event("pick_db", {"db": db})
    return {**state, "db": db}

def _mcp_call_sql(db: str, query: str, mcp_cmd: str):
    req = {"tool": "sql.readonly", "db": db, "query": query}
    try:
        proc = subprocess.Popen(
            shlex.split(mcp_cmd),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = proc.communicate(input=json.dumps(req), timeout=120)
        if err:
            print("[MCP][stderr]", err)
        try:
            res = json.loads(out or "{}")
        except Exception as je:
            res = {"ok": False, "error": f"Salida MCP no es JSON válido: {je}", "raw": out}
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}

def node_sql_exec(state: POIState):
    if not state["sql"]:
        return {**state, "reply": state.get("reply") or "No hay SQL para ejecutar."}

    res = _mcp_call_sql(state["db"], state["sql"], mcp_cmd)
    if not res.get("ok"):
        audit_event("sql_exec_error", {"db": state["db"], "error": res.get("error", "")})
        return {**state, "reply": f"Error al ejecutar SQL en {state['db']}: {res.get('error','')}", "sql": state["sql"]}

    cols = res.get("columns", []) or []
    rows = res.get("rows", []) or []

    # Si por alguna razón siguen viniendo COL1.., remap con OUTPUT_WHITELIST
    cols, rows = _maybe_remap_cols(cols, rows)

    # Guardrails de salida
    wl = get_whitelist_from_env()
    rows = whitelist_columns(rows, wl)
    rows = limit_rows(rows)

    audit_event("sql_exec_ok", {"db": state["db"], "rows": len(rows), "cols": len(cols)})
    return {**state, "cols": cols, "rows": rows, "sql": state["sql"]}

def node_chat(state: POIState):
    rag = state.get("rag_ctx") or "(sin documentos)"
    ctx = state.get("chat_ctx") or "(sin contexto)"
    raw = (chat_prompt | llm_chat | StrOutputParser()).invoke({
        "system": SYSTEM_PROMPT.format(target_table=target_table),
        "rag": rag,
        "ctx": ctx,
        "message": state["message"]
    })
    audit_event("chat", {"len": len(raw or "")})
    return {**state, "reply": raw}

# =========================
# Grafo
# =========================
graph = StateGraph(POIState)
graph.add_node("router", node_router)
graph.add_node("plan_sql", node_plan_sql)
graph.add_node("pick_db", node_pick_db)
graph.add_node("sql_exec", node_sql_exec)
graph.add_node("chat", node_chat)

def route_decider(state: POIState):
    return "plan_sql" if state["intent"] == "SQL" else "chat"

graph.set_entry_point("router")
graph.add_conditional_edges("router", route_decider, {"plan_sql": "plan_sql", "chat": "chat"})
graph.add_edge("plan_sql", "pick_db")
graph.add_edge("pick_db", "sql_exec")
graph.add_edge("sql_exec", END)
graph.add_edge("chat", END)

app = graph.compile()

# =========================
# Historial simple (UI)
# =========================
if "history" not in st.session_state:
    st.session_state["history"] = []

# =========================
# Render del chat
# =========================
st.subheader("Chat")
for role, content in st.session_state["history"]:
    with st.chat_message(role):
        if role == "assistant" and isinstance(content, dict) and content.get("type") == "table":
            st.success(f"Filas: {content.get('rows_count',0)} | Columnas: {len(content.get('cols',[]))}")
            df = pd.DataFrame(content.get("rows", []), columns=content.get("cols", []))
            st.dataframe(df.head(50), use_container_width=True)
        else:
            st.markdown(content if isinstance(content, str) else str(content))

prompt = st.chat_input("Escribe tu mensaje...")
if prompt:
    st.session_state["history"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    state_in: POIState = {
        "message": prompt,
        "intent": "",
        "sql": "",
        "db": "",
        "rows": [],
        "cols": [],
        "reply": "",
        "chat_ctx": "",
        "rag_ctx": "",
    }
    state_out = app.invoke(state_in)

    if state_out.get("rows"):
        if show_sql_debug and state_out.get("sql"):
            with st.chat_message("assistant"):
                st.markdown("**SQL generada:**")
                st.code(state_out["sql"], language="sql")
        with st.chat_message("assistant"):
            df = pd.DataFrame(state_out["rows"], columns=state_out.get("cols", []))
            st.success(f"OK. Filas: {len(df)} | Columnas: {len(df.columns)}")
            st.dataframe(df.head(50), use_container_width=True)
        st.session_state["history"].append((
            "assistant",
            {"type": "table", "rows_count": len(state_out["rows"]), "rows": state_out["rows"], "cols": state_out.get("cols", [])}
        ))
        summary = f"Consulta ejecutada en **{state_out.get('db','prd').upper()}**. Filas: **{len(state_out['rows'])}**."
        with st.chat_message("assistant"):
            st.markdown(summary)
        st.session_state["history"].append(("assistant", summary))
    else:
        reply = state_out.get("reply") or "Sin respuesta."
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state["history"].append(("assistant", reply))
