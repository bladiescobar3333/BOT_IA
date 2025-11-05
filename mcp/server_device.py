# mcp/sp_client_device.py
from io import BytesIO
import pandas as pd
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.device_code_provider import DeviceCodeProvider

def read_excel_from_sharepoint_device(site_url: str, server_relative_path: str):
    """
    Autenticación por Device Code (interactiva).
    - Abre microsoft.com/devicelogin
    - Ingresa el código mostrado en consola
    - Inicia con tu cuenta corporativa
    """
    print("\n🔐 Iniciando sesión con Microsoft (Device Code Login)...")
    ctx = ClientContext(site_url).with_credentials(DeviceCodeProvider(site_url))
    file = ctx.web.get_file_by_server_relative_url(server_relative_path)
    content = file.download().execute_query().content

    df = pd.read_excel(BytesIO(content))
    cols = [str(c) for c in df.columns]
    rows = df.where(pd.notnull(df), None).to_dict(orient="records")
    print(f"✅ Archivo leído: {len(rows)} filas, {len(cols)} columnas")
    return cols, rows
