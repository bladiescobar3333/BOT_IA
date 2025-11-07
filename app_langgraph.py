import os, sys, json, shlex, subprocess, re
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from typing import Literal, Dict, Any, TypedDict, List, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from common.prompts import SYSTEM_PROMPT, INTENT_TEMPLATE, SQL_TEMPLATE, CHAT_TEMPLATE
from security.guardrails_input import sanitize_user_input
from security.guardrails_sql import secure_sql_pipeline
from security.guardrails_output import limit_rows, whitelist_columns, get_whitelist_from_env
from observability.langsmith_obs import setup_langsmith, audit_event

# Import perezoso/seguro de RAG
RAG_OK = True
try:
    from rag_utils import index_dataframe, query_context
except Exception as e:
    RAG_OK = False
    index_dataframe = query_context = None



# =========================
# .env + LangSmith
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
        test_db = st.selectbox("DSN a probar", ["prd", "cfa"], index=0)
        if st.button("Probar MCP", use_container_width=True):
            test_req = {"tool": "sql.ping", "db": test_db}
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
                    msg = res.get("message", "Conexión OK")
                    dsn_used = res.get("dsn_used", "")
                    st.success(f"✅ MCP OK ({test_db}): {msg}\n\nDSN usado: {dsn_used[:500]}")
                else:
                    st.error(f"❌ MCP falló en {test_db}: {res.get('error', 'sin detalle')}")
            except Exception as e:
                st.error(f"❌ No se pudo ejecutar MCP: {e}")

    with c2:
        st.caption("Tip: Mantén este comando; ya utiliza tu venv correctamente.")

    # ===== NUEVO BLOQUE RAG =====
    st.header("💾 RAG (opcional)")
    use_rag = st.checkbox("Usar RAG (activar indexación de resultados grandes)", value=False)
    # Si rag_utils no cargó, desactiva el toggle
    if use_rag and not RAG_OK:
        st.warning("RAG no disponible: corrige dependencias/archivo rag_utils.py. Desactivando…")
        use_rag = False
    # Si no hay clave, desactiva el toggle
    if use_rag and not (google_key or "").strip():
        st.warning("Falta GOOGLE_API_KEY para embeddings. Desactivando RAG…")
        use_rag = False

    rag_threshold = st.number_input("Umbral de filas para indexar", min_value=100, max_value=500000, value=5000, step=500)
    rag_chunk = st.number_input("Tamaño de chunk", min_value=200, max_value=4000, value=1000, step=100)
    rag_overlap = st.number_input("Solape de chunk", min_value=0, max_value=1000, value=100, step=50)
    rag_topk = st.slider("Top-K recuperación", min_value=2, max_value=20, value=8)

st.caption("Tip: 'suma DESCARG CALLAO 2024 en prd', 'total DECLARA VEGUETA 2023 en qa', 'Qué es RAP'.")



# =========================
# LLM helpers
# =========================
def make_llm(temperature: float = 0.2):
    if not (google_key or "").strip():
        st.warning("Falta GOOGLE_API_KEY.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature, api_key=google_key)


llm_router = make_llm(0.1)
llm_chat = make_llm(0.2)

intent_prompt = PromptTemplate.from_template(INTENT_TEMPLATE)
sql_prompt = PromptTemplate.from_template(SQL_TEMPLATE)
chat_prompt = PromptTemplate.from_template(CHAT_TEMPLATE)


# =========================
# Estado
# =========================
class POIState(TypedDict):
    message: str
    intent: Literal["SQL", "CHAT", ""]
    sql: str
    db: Literal["prd", "qa", "his", "cfa", ""]
    rows: List[Dict[str, Any]]
    cols: List[str]
    reply: str
    chat_ctx: str
    rag_ctx: str


# =========================
# Helpers SQL Server
# =========================
def normalize_sqlserver(sql: str) -> str:
    s = (sql or "").strip()
    s = re.sub(r";\s*$", "", s, flags=re.IGNORECASE)

    m_off_lim = re.search(r"\bOFFSET\s+(\d+)\s+LIMIT\s+(\d+)\b", s, flags=re.IGNORECASE)
    m_lim_off = re.search(r"\bLIMIT\s+(\d+)\s+OFFSET\s+(\d+)\b", s, flags=re.IGNORECASE)
    if m_off_lim or m_lim_off:
        if m_off_lim:
            off = int(m_off_lim.group(1))
            lim = int(m_off_lim.group(2))
            s = re.sub(r"\bOFFSET\s+\d+\s+LIMIT\s+\d+\b", "", s, flags=re.IGNORECASE)
        else:
            lim = int(m_lim_off.group(1))
            off = int(m_lim_off.group(2))
            s = re.sub(r"\bLIMIT\s+\d+\s+OFFSET\s+\d+\b", "", s, flags=re.IGNORECASE)
        if not re.search(r"\bORDER\s+BY\b", s, flags=re.IGNORECASE):
            s += " ORDER BY FECHA DESC"
        s += f" OFFSET {off} ROWS FETCH NEXT {lim} ROWS ONLY"
        return s

    if re.search(r"^\s*SELECT\s+TOP\s+\d+\b", s, flags=re.IGNORECASE):
        return s

    m_any_lim = re.search(r"\b(LIMIT|LIMTI)\s+(\d+)\b", s, flags=re.IGNORECASE)
    if m_any_lim:
        n = int(m_any_lim.group(2))
        s = re.sub(r"\b(LIMIT|LIMTI)\s+\d+\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*SELECT\s+", f"SELECT TOP {n} ", s, flags=re.IGNORECASE)
        return s

    return s


def strip_auto_aliases(sql: str) -> str:
    return re.sub(r"(?i)\s+AS\s+COL\d+\b", "", sql or "")


def sanitize_sql_for_exec(sql: str) -> str:
    s = (sql or "").strip()
    s = re.sub(r";\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# =========================
# NEW — Mejoras SQL
# =========================
def _normalize_planta_equals_to_like(sql: str) -> str:
    """
    Convierte:
      Z.PLANTA = 'VEGUETA' → UPPER(LTRIM(RTRIM(Z.PLANTA))) LIKE '%VEGUETA%'
      PD.PLANTA = 'CALLAO' → UPPER(LTRIM(RTRIM(PD.PLANTA))) LIKE '%CALLAO%'
    """
    def repl_z(m):
        value = m.group(1).strip("'").strip()
        return f"UPPER(LTRIM(RTRIM(Z.PLANTA))) LIKE '%{value.upper()}%'"

    def repl_pd(m):
        value = m.group(1).strip("'").strip()
        return f"UPPER(LTRIM(RTRIM(PD.PLANTA))) LIKE '%{value.upper()}%'"

    # específicos por alias
    sql = re.sub(r"(?i)Z\.PLANTA\s*=\s*'([^']+)'", repl_z, sql)
    sql = re.sub(r"(?i)PD\.PLANTA\s*=\s*'([^']+)'", repl_pd, sql)

    # genérico (si el prompt generó solo PLANTA = 'X'), asumimos Z por defecto
    sql = re.sub(r"(?i)\bPLANTA\s*=\s*'([^']+)'", repl_z, sql)
    return sql


def _ensure_top1_when_last(sql: str, user_msg: str) -> str:
    """Si el usuario pide 'última/último', fuerza TOP 1 y ORDER BY DESC."""
    msg = user_msg.lower()
    if not any(w in msg for w in ("ultima", "última", "ultimo", "último")):
        return sql
    s = sql
    if not re.search(r"(?i)\bSELECT\s+TOP\s+\d+\b", s):
        s = re.sub(r"(?i)^\s*SELECT\s+", "SELECT TOP 1 ", s, count=1)
    if not re.search(r"(?i)\bORDER\s+BY\b", s):
        if "Z.FECHA" in s:
            s += " ORDER BY Z.FECHA DESC"
        elif "PD.FECHA_PD" in s:
            s += " ORDER BY PD.FECHA_PD DESC"
        else:
            s += " ORDER BY FECHA DESC"
    return s


def _tune_sql_for_intent(sql: str, user_msg: str) -> str:
    s = _normalize_planta_equals_to_like(sql)
    s = _ensure_top1_when_last(s, user_msg)
    return s


def _maybe_remap_cols(cols: List[str], rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not cols:
        return cols, rows
    if all(re.fullmatch(r"(?i)COL\d+", (c or "")) for c in cols):
        wl = get_whitelist_from_env()
        if wl and len(wl) >= len(cols):
            new_cols = wl[:len(cols)]
            new_rows = [{new_cols[i]: r.get(cols[i]) for i in range(len(cols))} for i, r in enumerate(rows)]
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

    # --- Heurística de corrección (flip a SQL si el texto parece consulta)
    msg = clean.lower()
    query_cues = [
        "select", "última", "ultima", "último", "ultimo", "total", "suma",
        "sumar", "promedio", "avg", "contar", "count", "filtra", "filtro",
        "por planta", "por embarca", "por flota", "join", "plan vs real"
    ]
    column_cues = [
        "marea", "planta", "fecha", "flota", "embarca", "declara", "descarg",
        "fecha_pd", "orden_pd", "linea_pd", "vel_pd", "espera_pd", "declara_pd"
    ]
    alias_cues = [" z.", " pd.", "z.", "pd."]

    looks_like_sql = any(w in msg for w in query_cues + column_cues) or any(a in msg for a in alias_cues)
    if intent == "CHAT" and looks_like_sql:
        intent = "SQL"

    audit_event("router", {"message": clean, "intent": intent, "llm_intent": lab})
    return {**state, "message": clean, "intent": intent}


def node_plan_sql(state: POIState):
    raw = (sql_prompt | llm_router | StrOutputParser()).invoke({
        "system": SYSTEM_PROMPT.format(target_table=target_table),
        "target_table": target_table,
        "question": state["message"]
    })
    ok, why, sql = secure_sql_pipeline(raw)
    if ok:
        sql = _tune_sql_for_intent(sql, state["message"])  # ✅ Mejora planta + TOP 1
        sql = normalize_sqlserver(sql)
        sql = strip_auto_aliases(sql)
        sql = sanitize_sql_for_exec(sql)
    audit_event("plan_sql", {"ok": ok, "why": why, "sql": (sql or "")[:400]})
    if not ok:
        return {**state, "sql": "", "reply": f"SQL inválida: {why}"}
    return {**state, "sql": sql}


def node_pick_db(state: POIState):
    m = state["message"].lower()
    db = "prd"
    if " en qa" in m:
        db = "qa"
    if " en his" in m or " en hist" in m:
        db = "his"
    if " en cfa" in m or " en repl" in m:
        db = "cfa"

    # si no especifican, heurística: consultas de plan (PD) van a cfa
    if db == "prd":
        if any(t in m for t in (" plan ", "plandescarga", "planificado", "join", "plan vs real")):
            db = "cfa"

    sql_low = (state.get("sql") or "").lower()
    if "plandescargaoutput" in sql_low or re.search(r"\bpd\.", sql_low):
        db = "cfa"

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

    clean_sql = sanitize_sql_for_exec(normalize_sqlserver(state["sql"]))

    def _exec(dbname: str):
        return dbname, _mcp_call_sql(dbname, clean_sql, mcp_cmd)

    db_used, res = _exec(state["db"])

    err_msg = (res.get("error") or "").lower()
    if (not res.get("ok")) and ("invalid object name" in err_msg or "plandescargaoutput" in err_msg) and db_used != "cfa":
        audit_event("sql_exec_retry_cfa", {"from": db_used})
        db_used, res = _exec("cfa")

    if not res.get("ok"):
        audit_event("sql_exec_error", {"db": db_used, "error": res.get("error", "")})
        return {**state, "db": db_used, "reply": f"Error {db_used}: {res.get('error','')}\nSQL: {clean_sql}", "sql": clean_sql}

    cols = res.get("columns", []) or []
    rows = res.get("rows", []) or []
    cols, rows = _maybe_remap_cols(cols, rows)
    wl = get_whitelist_from_env()
    rows = whitelist_columns(rows, wl)
    rows = limit_rows(rows)

    # === NEW: indexación automática con RAG si resultado es grande ===
    rag_ctx_val = state.get("rag_ctx") or ""
    try:
        if use_rag and len(rows) >= rag_threshold:
            df_full = pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows)
            ns = index_dataframe(
                df=df_full,
                sql=clean_sql,
                db=db_used,
                google_key=google_key,
                chunk_size=int(rag_chunk),
                chunk_overlap=int(rag_overlap),
            )
            # Recuperamos un contexto base alineado al último prompt del usuario
            rag_ctx_val = query_context(
                user_query=state.get("message", ""),
                namespace=ns,
                google_key=google_key,
                k=int(rag_topk),
            )
            audit_event("rag_index_ok", {"ns": ns, "rows": len(rows), "topk": int(rag_topk)})
    except Exception as e:
        audit_event("rag_index_error", {"err": str(e)})

    audit_event("sql_exec_ok", {"db": db_used, "rows": len(rows), "cols": len(cols)})
    return {**state, "db": db_used, "cols": cols, "rows": rows, "sql": clean_sql, "rag_ctx": rag_ctx_val}

def node_chat(state: POIState):
    # Si el usuario marca RAG pero viene por camino CHAT (sin SQL), intentamos recuperar
    rag = state.get("rag_ctx") or ""
    if use_rag and not rag:
        try:
            # Heurística: usa el último namespace indexado si lo guardas en sesión.
            # Para simplificar, intentamos recuperar de "cualquier" colección creada recientemente no es trivial;
            # aquí solo usamos el rag_ctx si ya lo llenó node_sql_exec. (Suficiente para empezar)
            pass
        except Exception as e:
            audit_event("chat_rag_skip", {"err": str(e)})

    ctx = state.get("chat_ctx") or "(sin contexto)"
    raw = (chat_prompt | llm_chat | StrOutputParser()).invoke({
        "system": SYSTEM_PROMPT.format(target_table=target_table),
        "rag": rag if rag else "(sin documentos)",
        "ctx": ctx,
        "message": state["message"]
    })
    audit_event("chat", {"len": len(raw or "")})
    return {**state, "reply": raw}


# =========================
# Grafo LangGraph
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
# Render del chat (UI)
# =========================
if "history" not in st.session_state:
    st.session_state["history"] = []

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
    # 1) Guardar mensaje del usuario
    st.session_state["history"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Ejecutar grafo
    state_in: POIState = {
        "message": prompt, "intent": "", "sql": "", "db": "",
        "rows": [], "cols": [], "reply": "", "chat_ctx": "", "rag_ctx": "",
    }
    state_out = app.invoke(state_in)

    # 3) Render y persistencia de la respuesta del asistente
    assistant_blocks: list = []  # para almacenar lo que mostraremos y luego persistir

    intent_lbl = f"Intención: **{state_out.get('intent','')}**"
    if state_out.get("db"):
        intent_lbl += f" · DB: **{state_out['db'].upper()}**"
    assistant_blocks.append(intent_lbl)

    if show_sql_debug and state_out.get("sql"):
        assistant_blocks.append("**SQL generada:**\n```sql\n" + state_out["sql"] + "\n```")

    cols = state_out.get("cols") or []
    rows = state_out.get("rows") or []
    reply = (state_out.get("reply") or "").strip()

    with st.chat_message("assistant"):
        st.caption(intent_lbl)

        if show_sql_debug and state_out.get("sql"):
            st.markdown("**SQL generada:**")
            st.code(state_out["sql"], language="sql")

        if cols:
            df = pd.DataFrame(rows, columns=cols)
            st.success(f"Resultado — Filas: {len(rows)} | Columnas: {len(cols)}")
            if len(rows) == 0:
                st.info("La consulta ejecutó correctamente pero no devolvió filas.")
            st.dataframe(df.head(50), use_container_width=True)

            # Persistir como bloque tipo tabla
            st.session_state["history"].append((
                "assistant",
                {"type": "table", "rows": rows, "cols": cols, "rows_count": len(rows)}
            ))
        else:
            if reply:
                if reply.lower().startswith("error"):
                    st.error(reply)
                else:
                    st.markdown(reply)
                st.session_state["history"].append(("assistant", reply))
            else:
                msg = "No hay datos para mostrar. Revisa la SQL generada o intenta otra consulta."
                st.warning(msg)
                st.session_state["history"].append(("assistant", msg))
