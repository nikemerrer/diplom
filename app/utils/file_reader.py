# app/utils/file_reader.py
# -*- coding: utf-8 -*-
import io
import os
import tempfile
import subprocess
from typing import Optional

# ---------- опциональные зависимости ----------
try:
    from docx import Document  # type: ignore
except Exception:
    Document = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    pdfminer_extract_text = None  # type: ignore

try:
    from striprtf.striprtf import rtf_to_text  # type: ignore
except Exception:
    rtf_to_text = None  # type: ignore


# ---------- утилиты ----------
def _decode_txt_best_effort(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")


def _is_pdf(b: bytes) -> bool:
    return b.startswith(b"%PDF-")

def _is_docx(b: bytes) -> bool:
    # zip magic
    return b.startswith(b"PK\x03\x04")

def _is_rtf(b: bytes) -> bool:
    return b.lstrip().startswith(b"{\\rtf")

def _is_doc_ole(b: bytes) -> bool:
    # OLE header for .doc, xls, etc.
    return b.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1")


# ---------- отдельные парсеры форматов ----------
def _read_docx(b: bytes) -> str:
    if Document is None:
        raise RuntimeError("python-docx не установлен. Добавь 'python-docx' в requirements.txt")
    f = io.BytesIO(b)
    doc = Document(f)
    lines = []
    for p in doc.paragraphs:
        txt = (p.text or "").strip()
        if txt:
            lines.append(txt)
    for table in getattr(doc, "tables", []):
        for row in table.rows:
            cells = []
            for cell in row.cells:
                val = (cell.text or "").strip()
                if val:
                    cells.append(val)
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def _read_pdf_with_pypdf2(b: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 не установлен. Добавь 'PyPDF2' в requirements.txt")
    reader = PdfReader(io.BytesIO(b))
    chunks = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            chunks.append(t)
    return "\n".join(chunks).strip()


def _read_pdf_with_pdfminer(b: bytes) -> str:
    if pdfminer_extract_text is None:
        raise RuntimeError("pdfminer.six не установлен. Добавь 'pdfminer.six' в requirements.txt")
    return (pdfminer_extract_text(io.BytesIO(b)) or "").strip()


def _read_pdf(b: bytes) -> str:
    # Сначала быстрее (PyPDF2), затем надёжнее (pdfminer)
    if PdfReader is not None:
        try:
            txt = _read_pdf_with_pypdf2(b)
            if txt.strip():
                return txt
        except Exception:
            pass
    if pdfminer_extract_text is not None:
        try:
            txt = _read_pdf_with_pdfminer(b)
            if txt.strip():
                return txt
        except Exception:
            pass
    # Вернём подсказку: возможно, это скан (нужен OCR)
    raise RuntimeError("Не удалось извлечь текст из PDF (возможно, это скан без текста).")


def _read_rtf(b: bytes) -> str:
    if rtf_to_text is None:
        raise RuntimeError("striprtf не установлен. Добавь 'striprtf' в requirements.txt")
    try:
        return rtf_to_text(_decode_txt_best_effort(b)).strip()
    except Exception:
        return ""


def _read_doc_with_antiword(tmp_path: str) -> str:
    # -w 0 => без переноса строк; -m UTF-8 => кодировка вывода
    proc = subprocess.run(
        ["antiword", "-w", "0", "-m", "UTF-8", tmp_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout.decode("utf-8", errors="ignore").strip()
    return ""


def _read_doc_with_catdoc(tmp_path: str) -> str:
    # catdoc по умолчанию в CP1251/KOI8, попросим UTF-8, если поддерживается
    proc = subprocess.run(
        ["catdoc", "-d", "utf-8", tmp_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout:
        try:
            return proc.stdout.decode("utf-8", errors="ignore").strip()
        except Exception:
            pass
    # fallback: как есть
    if proc.stdout:
        return proc.stdout.decode("latin-1", errors="ignore").strip()
    return ""


def _read_doc(b: bytes) -> str:
    """
    Старый .doc (OLE). Используем external tools: antiword -> catdoc.
    """
    with tempfile.NamedTemporaryFile(suffix=".doc", delete=True) as tf:
        tf.write(b)
        tf.flush()
        text = ""
        # 1) antiword
        try:
            text = _read_doc_with_antiword(tf.name)
        except FileNotFoundError:
            # antiword не установлен
            text = ""
        except Exception:
            text = text or ""
        # 2) catdoc (если antiword не дал результат)
        if not text:
            try:
                text = _read_doc_with_catdoc(tf.name)
            except FileNotFoundError:
                text = ""
            except Exception:
                text = text or ""
        return text.strip()


# ---------- публичные entrypoints ----------
def read_file_to_text(file_bytes: bytes, filename: Optional[str] = None) -> str:
    fname = (filename or "").lower()
    ext = os.path.splitext(fname)[1]

    # Если расширение известно — используем по нему
    if ext == ".docx":
        return _read_docx(file_bytes)
    if ext == ".pdf":
        return _read_pdf(file_bytes)
    if ext == ".rtf":
        return _read_rtf(file_bytes)
    if ext == ".doc":
        return _read_doc(file_bytes)
    if ext == ".txt" or ext == "":
        # .txt или неизвестно — попробуем сигнатуры и/или текст
        b = file_bytes
        if _is_pdf(b):
            return _read_pdf(b)
        if _is_docx(b):
            return _read_docx(b)
        if _is_rtf(b):
            return _read_rtf(b)
        if _is_doc_ole(b):
            return _read_doc(b)
        return _decode_txt_best_effort(b).strip()

    # Неизвестное расширение — попробуем сигнатуры, затем текст
    b = file_bytes
    if _is_pdf(b):
        return _read_pdf(b)
    if _is_docx(b):
        return _read_docx(b)
    if _is_rtf(b):
        return _read_rtf(b)
    if _is_doc_ole(b):
        return _read_doc(b)
    return _decode_txt_best_effort(b).strip()


# Синонимы для совместимости
def read_bytes_to_text(file_bytes: bytes, filename: Optional[str] = None) -> str:
    return read_file_to_text(file_bytes, filename=filename)

def get_text_from_file(file_bytes: bytes, filename: Optional[str] = None) -> str:
    return read_file_to_text(file_bytes, filename=filename)
