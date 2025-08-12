# app_streamlit_cefr_uploader.py
# -------------------------------
# CEFR Uploader & Analyzer (PDF/EPUB) + opcional Google Drive/Sheets
# Autocontido. Nenhum import de outros arquivos.

import io
import re
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st

# ======= EXTRAÇÃO DE TEXTO =======

def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 25) -> Tuple[str, List[int]]:
    """Extrai texto e palavras/página de um PDF em memória."""
    try:
        from PyPDF2 import PdfReader
    except Exception:
        st.error("Instale PyPDF2: pip install PyPDF2")
        return "", []
    text = []
    words_per_page = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = len(reader.pages)
        for i in range(min(pages, max_pages)):
            p = reader.pages[i]
            t = p.extract_text() or ""
            text.append(t)
            w = re.findall(r"[A-Za-z']+", t)
            words_per_page.append(len(w))
    except Exception as e:
        st.warning(f"Falha ao extrair PDF: {e}")
    return "\n".join(text), words_per_page

def extract_text_from_epub(file_bytes: bytes, max_items: int = 200) -> Tuple[str, List[int]]:
    """Extrai texto de um EPUB (aproxima páginas por item)."""
    try:
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except Exception:
        st.error("Instale ebooklib e beautifulsoup4: pip install ebooklib beautifulsoup4")
        return "", []
    text = []
    wpp = []
    try:
        book = epub.read_epub(io.BytesIO(file_bytes))
        count = 0
        for item in book.get_items():
            if item.get_type() == 9:  # DOCUMENT
                html = item.get_content()
                soup = BeautifulSoup(html, "html.parser")
                t = soup.get_text(" ", strip=True)
                text.append(t)
                w = re.findall(r"[A-Za-z']+", t)
                wpp.append(len(w))
                count += 1
                if count >= max_items:
                    break
    except Exception as e:
        st.warning(f"Falha ao extrair EPUB: {e}")
    return "\n".join(text), wpp

# ======= MÉTRICAS / REGRAS =======

ACTIVITY_KEYWORDS = [
    "activity", "activities", "puzzle", "maze", "color", "colour", "draw", "trace",
    "spot the", "find the", "join the dots", "dot-to-dot", "answers", "solutions",
    "stickers", "search-and-find", "sudoku", "grid", "decorate", "shadow", "match",
    "labyrinth", "wordsearch", "toy-doku", "crossword"
]

def detect_activity_hits(text: str) -> int:
    t = text.lower()
    return sum(1 for kw in ACTIVITY_KEYWORDS if kw in t)

def count_syllables(word: str) -> int:
    w = word.lower()
    vowels = "aeiouy"
    cnt, prev = 0, False
    for ch in w:
        if ch in vowels:
            if not prev:
                cnt += 1
            prev = True
        else:
            prev = False
    if w.endswith("e"):
        cnt = max(1, cnt - 1)
    return cnt or 1

def _tokenize(text: str):
    return re.findall(r"[A-Za-z']+", text)

def flesch_reading_ease(text: str) -> float:
    sents = re.split(r'[.!?]+', text)
    sents = [s for s in sents if s.strip()]
    words = _tokenize(text)
    if not words or not sents:
        return 0.0
    syll = sum(count_syllables(w) for w in words)
    return round(206.835 - 1.015*(len(words)/len(sents)) - 84.6*(syll/len(words)), 2)

def flesch_kincaid_grade(text: str) -> float:
    sents = re.split(r'[.!?]+', text)
    sents = [s for s in sents if s.strip()]
    words = _tokenize(text)
    if not words or not sents:
        return 0.0
    syll = sum(count_syllables(w) for w in words)
    return round(0.39*(len(words)/len(sents)) + 11.8*(syll/len(words)) - 15.59, 2)

def avg_sentence_len(text: str) -> float:
    sents = re.split(r'[.!?]+', text)
    sents = [s for s in sents if s.strip()]
    words = _tokenize(text)
    if not sents:
        return 0.0
    return round(len(words)/len(sents), 2)

def pct_polysyllables(text: str) -> float:
    words = _tokenize(text)
    if not words:
        return 0.0
    polys = sum(1 for w in words if count_syllables(w) >= 3)
    return round(100*polys/len(words), 2)

def estimate_cefr(fk_grade: float, fre: float, asl: float, poly: float) -> str:
    # base
    if fk_grade <= 3.5:
        level = "A1"
    elif fk_grade <= 6:
        level = "A2"
    elif fk_grade <= 8:
        level = "B1"
    elif fk_grade <= 10:
        level = "B2"
    elif fk_grade <= 12:
        level = "C1"
    else:
        level = "C2"
    # ajustes
    if level in ("A2", "B1") and (asl > 20 and poly > 12):
        level = "B1" if level == "A2" else "B2"
    if level in ("B2", "C1") and fre > 65:
        level = "B1" if level == "B2" else "B2"
    return level

def classify_book(sample_text: str, words_per_page: List[int]) -> Dict:
    mean_wpp = round(sum(words_per_page)/max(1, len(words_per_page)), 1) if words_per_page else 0.0
    med_wpp = float(pd.Series(words_per_page).median()) if words_per_page else 0.0
    hits = detect_activity_hits(sample_text)

    # Heurística de Atividades/Ilustrado
    low_density = (mean_wpp < 40) or (med_wpp < 30)
    is_activity = low_density or (hits >= 3)

    if is_activity:
        return {
            "tipo_livro": "Atividades/Ilustrado",
            "nivel_ingles_cefr": "Pré-A1 / N/A",
            "fre": None,
            "fk_grade": None,
            "avg_sentence_len": None,
            "pct_polysyllables": None,
            "palavras_por_pagina_media": mean_wpp,
            "palavras_por_pagina_mediana": med_wpp,
            "sinais_atividade_hits": hits,
            "justificativa_cefr": (
                f"Baixa densidade textual (média={mean_wpp}; mediana={med_wpp}) "
                f"e/ou padrão de atividades (hits={hits}). CEFR padrão não aplicável."
            ),
        }

    # Texto contínuo → calcular métricas
    fre = flesch_reading_ease(sample_text)
    fk = flesch_kincaid_grade(sample_text)
    asl = avg_sentence_len(sample_text)
    poly = pct_polysyllables(sample_text)
    level = estimate_cefr(fk, fre, asl, poly)

    return {
        "tipo_livro": "Texto contínuo",
        "nivel_ingles_cefr": level,
        "fre": fre,
        "fk_grade": fk,
        "avg_sentence_len": asl,
        "pct_polysyllables": poly,
        "palavras_por_pagina_media": mean_wpp,
        "palavras_por_pagina_mediana": med_wpp,
        "sinais_atividade_hits": hits,
        "justificativa_cefr": (
            f"Texto contínuo. Flesch={fre}, FK={fk}, sent/avg={asl}, polissilábicas={poly}%. "
            f"Densidade: média={mean_wpp} / mediana={med_wpp} palavras/página."
        ),
    }

def guess_isbn_from_name(name: str) -> str:
    digits = re.sub(r"\D", "", name or "")
    return digits if 10 <= len(digits) <= 13 else ""

APP_VERSION = "0.5.0"

# ======= GOOGLE INTEGRAÇÕES (opcional) =======

def load_google_services(sa_json_bytes: bytes):
    """Cria clientes do Drive e Sheets com credenciais de Service Account."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        import gspread
    except Exception:
        st.warning("Para integrações Google, instale: google-api-python-client google-auth gspread")
        return None, None, None

    scopes = [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    info = json.loads(sa_json_bytes.decode("utf-8"))
    creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)

    drive = build("drive", "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds)
    gc = gspread.authorize(creds)
    return drive, sheets, gc

def list_drive_pdfs_epubs(drive, folder_id: str):
    """Lista PDFs/EPUBs em uma pasta do Drive."""
    q = f"'{folder_id}' in parents and (mimeType='application/pdf' or mimeType='application/epub+zip') and trashed=false"
    results = drive.files().list(q=q, fields="files(id, name, mimeType)").execute()
    return results.get("files", [])

def download_drive_file(drive, file_id: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload
    req = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    return fh.getvalue()

def append_df_to_sheet(sheets, sheet_id: str, worksheet_name: str, df: pd.DataFrame):
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    rng = f"{worksheet_name}!A1"
    body = {"values": values}
    sheets.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=rng,
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()

# ======= UI =======

st.set_page_config(page_title="CEFR Uploader", page_icon="📚", layout="wide")

with st.sidebar:
    st.header("⚙️ Configuração")
    st.caption(f"Versão do app: {APP_VERSION}")

    st.subheader("Integrações Google (opcional)")
    sa_file = st.file_uploader("Service Account JSON", type=["json"])
    folder_id = st.text_input("Drive Folder ID (opcional)", "")
    sheet_id_results = st.text_input("Google Sheet ID (Resultados - opcional)", "")
    sheet_tab_results = st.text_input("Aba para Resultados (opcional)", "ResultadosCEFR")
    sheet_id_audit = st.text_input("Google Sheet ID (AuditLog - opcional)", "")
    sheet_tab_audit = st.text_input("Aba para AuditLog (opcional)", "AuditLog")

    enable_drive = st.checkbox("Usar pasta do Google Drive (processamento em lote)", value=False)
    list_btn = st.button("Listar arquivos da pasta (Drive)")
    process_drive_btn = st.button("Processar TODOS da pasta (Drive)")

st.title("📚 CEFR Uploader & Analyzer")
st.write("Envie **PDF/EPUB** para nivelamento automático em CEFR. Livros de atividades/ilustrados são detectados e marcados como **Pré‑A1 / N/A**.")

st.markdown("---")
files = st.file_uploader("📥 Envie seus arquivos PDF/EPUB", type=["pdf", "epub"], accept_multiple_files=True)
process_btn = st.button("▶️ Processar arquivos enviados", type="primary")

# Resultado mantido em memória na sessão
if "results_df" not in st.session_state:
    st.session_state["results_df"] = pd.DataFrame()

def analyze_binary(name: str, data: bytes) -> Dict:
    ext = (name.split(".")[-1] or "").lower()
    if ext == "pdf":
        sample_text, wpp = extract_text_from_pdf(data)
    elif ext == "epub":
        sample_text, wpp = extract_text_from_epub(data)
    else:
        return {"error": "Formato não suportado"}

    result = classify_book(sample_text, wpp)
    result["arquivo"] = name
    result["isbn"] = guess_isbn_from_name(name)
    result["data_processamento"] = datetime.utcnow().isoformat()
    result["versao_modelo"] = APP_VERSION
    return result

def results_to_df(results: List[Dict]) -> pd.DataFrame:
    cols = [
        "arquivo", "isbn", "tipo_livro", "nivel_ingles_cefr",
        "fre", "fk_grade", "avg_sentence_len", "pct_polysyllables",
        "palavras_por_pagina_media", "palavras_por_pagina_mediana",
        "sinais_atividade_hits", "justificativa_cefr",
        "versao_modelo", "data_processamento"
    ]
    df = pd.DataFrame(results)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

# --- Processamento local (arquivos enviados) ---
if process_btn and files:
    st.info(f"Processando {len(files)} arquivo(s)...")
    out = []
    pb = st.progress(0)
    for i, f in enumerate(files, 1):
        data = f.read()
        out.append(analyze_binary(f.name, data))
        pb.progress(i/len(files))
    df_out = results_to_df(out)
    st.session_state["results_df"] = pd.concat([st.session_state["results_df"], df_out], ignore_index=True)
    st.success("Concluído!")

# --- Integração Google: listar e/ou processar pasta do Drive ---
if (list_btn or process_drive_btn) and enable_drive:
    if not sa_file:
        st.error("Envie o Service Account JSON no sidebar.")
    elif not folder_id:
        st.error("Informe o Folder ID.")
    else:
        drive, sheets, gc = load_google_services(sa_file.read())
        if not drive:
            st.stop()
        files_drive = list_drive_pdfs_epubs(drive, folder_id)
        st.write(f"Arquivos na pasta: {len(files_drive)}")
        st.dataframe(pd.DataFrame(files_drive))

        if process_drive_btn and files_drive:
            st.info(f"Processando {len(files_drive)} arquivo(s) do Drive...")
            out = []
            pb = st.progress(0)
            for i, fmeta in enumerate(files_drive, 1):
                content = download_drive_file(drive, fmeta["id"])
                out.append(analyze_binary(fmeta["name"], content))
                pb.progress(i/len(files_drive))
            df_out = results_to_df(out)
            st.session_state["results_df"] = pd.concat([st.session_state["results_df"], df_out], ignore_index=True)
            st.success("Concluído!")

st.markdown("## 🧾 Resultados")
if not st.session_state["results_df"].empty:
    st.dataframe(st.session_state["results_df"], use_container_width=True, height=400)

    # Downloads
    csv_bytes = st.session_state["results_df"].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", data=csv_bytes, file_name="resultados_cefr.csv", mime="text/csv")

    # XLSX
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        st.session_state["results_df"].to_excel(writer, index=False, sheet_name="Resultados")
    st.download_button("⬇️ Baixar XLSX", data=xlsx_buf.getvalue(), file_name="resultados_cefr.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Gravar em Google Sheets (opcional)
    st.markdown("---")
    st.subheader("📤 Exportar para Google Sheets (opcional)")
    export_btn = st.button("Enviar resultados para Sheets")
    if export_btn:
        if not sa_file or not sheet_id_results or not sheet_tab_results:
            st.error("Preencha Service Account JSON, Sheet ID e Aba (Resultados).")
        else:
            drive, sheets, gc = load_google_services(sa_file.getvalue())
            if sheets:
                # escreve com cabeçalho + linhas (append)
                append_df_to_sheet(sheets, sheet_id_results, sheet_tab_results, st.session_state["results_df"])
                st.success("Resultados enviados para Google Sheets.")

    # AuditLog (opcional)
    st.subheader("🪵 Audit Log (opcional)")
    operador = st.text_input("Nome/E-mail do operador para log", "")
    audit_btn = st.button("Enviar AuditLog")
    if audit_btn:
        if not sa_file or not sheet_id_audit or not sheet_tab_audit:
            st.error("Preencha Service Account JSON, Sheet ID e Aba (AuditLog).")
        else:
            if not operador:
                st.warning("Informe o operador para registrar no log.")
            drive, sheets, gc = load_google_services(sa_file.getvalue())
            if sheets:
                df_log = st.session_state["results_df"].copy()
                df_log.insert(0, "operador", operador or "desconhecido")
                append_df_to_sheet(sheets, sheet_id_audit, sheet_tab_audit, df_log)
                st.success("AuditLog enviado.")

st.markdown("---")
st.caption("Dependências mínimas: streamlit PyPDF2 ebooklib beautifulsoup4 pandas xlsxwriter")
st.caption("Integrações Google (opcional): google-api-python-client google-auth gspread")
st.caption("Se 'streamlit' não estiver no PATH, rode: python -m streamlit run app_streamlit_cefr_uploader.py")
