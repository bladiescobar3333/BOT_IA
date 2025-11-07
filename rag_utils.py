# rag_utils.py
import os
from typing import List, Dict, Any, Optional

# === (OPCIONAL) Autenticación SharePoint ===
# Usa MSAL + Graph o app-only de office365-rest-python-client
# Este módulo NO ejecuta nada al importar.

def get_sharepoint_client(auth_mode: str = "app", **kwargs):
    """
    auth_mode: "app" (Client ID + Secret) o "device" (MFA interactivo).
    Devuelve un objeto cliente o lanza excepción descriptiva.
    """
    mode = (auth_mode or "app").lower()
    if mode == "app":
        # App-only (recomendado para servidores, no requiere MFA)
        from office365.sharepoint.client_context import ClientContext
        from office365.runtime.auth.client_credential import ClientCredential

        site_url = kwargs["site_url"]
        client_id = kwargs["client_id"]
        client_secret = kwargs["client_secret"]
        ctx = ClientContext(site_url).with_credentials(ClientCredential(client_id, client_secret))
        return ctx

    elif mode == "device":
        # Device Code (MFA interactivo)
        import msal
        from office365.graph_client import GraphClient

        tenant_id = kwargs["tenant_id"]
        client_id = kwargs["client_id"]
        scopes = ["https://graph.microsoft.com/.default"]

        app = msal.PublicClientApplication(client_id=client_id, authority=f"https://login.microsoftonline.com/{tenant_id}")
        flow = app.initiate_device_flow(scopes=scopes)
        if "user_code" not in flow:
            raise RuntimeError(f"No se pudo iniciar device code: {flow}")

        # Muestra las instrucciones al usuario (imprime; tú decides cómo mostrar en tu UI)
        print(f"Ve a {flow['verification_uri']} e ingresa el código: {flow['user_code']}")

        result = app.acquire_token_by_device_flow(flow)
        if "access_token" not in result:
            raise RuntimeError(f"Fallo al obtener token (device code): {result}")

        client = GraphClient(lambda: result["access_token"])
        return client

    else:
        raise ValueError("auth_mode inválido. Usa 'app' o 'device'.")

def download_sharepoint_file(ctx, server_relative_url: str) -> bytes:
    """
    Descarga un archivo de SharePoint dado un ClientContext (app-only) y una URL relativa.
    No se llama automáticamente: la app decide cuándo.
    """
    from office365.sharepoint.files.file import File
    resp = File.open_binary(ctx, server_relative_url)
    return resp.content

# === Motor RAG: NO depende de SharePoint. Sólo recibe datos/SQL ya calculados. ===
def index_dataframe(df, sql: str, db: str, google_key: Optional[str], chunk_size: int = 1000, chunk_overlap: int = 100) -> str:
    """
    Indexa el DF en tu backend de embeddings (implementación de ejemplo/no-op).
    Devuelve un 'namespace' lógico que luego usarás para recuperar contexto.
    """
    # Aquí pondrías tu Chroma/FAISS/Vertex/pgvector, etc.
    # Por ahora devolvemos un namespace sintético.
    ns = f"rag::{db}::rows={len(df)}::hash={abs(hash(sql))%10_000_000}"
    return ns

def query_context(user_query: str, namespace: str, google_key: Optional[str], k: int = 8) -> str:
    """
    Recupera k chunks relevantes del namespace. Versión placeholder que devuelve un texto corto.
    """
    return f"[RAG] namespace={namespace} · q='{user_query[:120]}' · topk={k}"
