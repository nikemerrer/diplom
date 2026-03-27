# app/utils/file_reader.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os
import tempfile
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    from pdf2image import convert_from_bytes  # type: ignore
except Exception:
    convert_from_bytes = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    pdfminer_extract_text = None  # type: ignore

try:
    from striprtf.striprtf import rtf_to_text  # type: ignore
except Exception:
    rtf_to_text = None  # type: ignore


def _decode_txt_best_effort(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")


def _decode_process_output(b: bytes) -> str:
    if not b:
        return ""
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return b.decode(enc, errors="ignore").strip()
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore").strip()


def _clean_lines(lines: List[str]) -> str:
    cleaned: List[str] = []
    for line in lines:
        value = (line or "").strip()
        if value:
            cleaned.append(value)
    return "\n".join(cleaned).strip()


def _is_pdf(b: bytes) -> bool:
    return b.startswith(b"%PDF-")


def _is_zip(b: bytes) -> bool:
    return b.startswith(b"PK\x03\x04") or b.startswith(b"PK\x05\x06") or b.startswith(b"PK\x07\x08")


def _is_rtf(b: bytes) -> bool:
    return b.lstrip().startswith(b"{\\rtf")


def _is_doc_ole(b: bytes) -> bool:
    return b.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1")


def _docx_package_has_document_xml(b: bytes) -> bool:
    if not _is_zip(b):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(b)) as zf:
            names = set(zf.namelist())
        return "word/document.xml" in names
    except Exception:
        return False


def _extract_text_from_xml(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    parts: List[str] = []
    for elem in root.iter():
        if elem.text:
            value = elem.text.strip()
            if value:
                parts.append(value)
    return _clean_lines(parts)


def _extract_lines_from_docx_obj(doc: Any) -> List[str]:
    lines: List[str] = []

    for p in getattr(doc, "paragraphs", []):
        value = (p.text or "").strip()
        if value:
            lines.append(value)

    for table in getattr(doc, "tables", []):
        for row in table.rows:
            cells: List[str] = []
            for cell in row.cells:
                value = (cell.text or "").strip()
                if value:
                    cells.append(value)
            if cells:
                lines.append(" | ".join(cells))

    return lines


def _read_docx(file_bytes: bytes) -> str:
    if Document is not None:
        try:
            doc = Document(io.BytesIO(file_bytes))
            text = _clean_lines(_extract_lines_from_docx_obj(doc))
            if text:
                return text
        except Exception:
            pass

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = [
                name for name in zf.namelist()
                if name.startswith("word/") and name.endswith(".xml")
            ]
            parts: List[str] = []
            for name in names:
                try:
                    xml_text = _extract_text_from_xml(zf.read(name))
                except Exception:
                    continue
                if xml_text:
                    parts.append(xml_text)
            return _clean_lines(parts)
    except Exception:
        return ""


def _read_pdf_with_pypdf2(b: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 не установлен")
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
        raise RuntimeError("pdfminer.six не установлен")
    return (pdfminer_extract_text(io.BytesIO(b)) or "").strip()


def _read_pdf(b: bytes) -> str:
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

    ocr_lang = os.getenv("OCR_LANG", "rus+eng")
    if convert_from_bytes is not None and pytesseract is not None:
        try:
            images = convert_from_bytes(b)
            pages = []
            for img in images:
                txt = pytesseract.image_to_string(img, lang=ocr_lang)  # type: ignore
                if txt:
                    pages.append(txt)
            if pages:
                return "\n".join(pages).strip()
        except Exception:
            pass

    raise RuntimeError("Не удалось извлечь текст из PDF")


def _read_rtf(b: bytes) -> str:
    if rtf_to_text is None:
        raise RuntimeError("striprtf не установлен")
    try:
        return rtf_to_text(_decode_txt_best_effort(b)).strip()
    except Exception:
        return ""


def _run_external_tool(args: List[str], timeout: int = 20) -> Dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")

    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        env=env,
    )
    return {
        "returncode": proc.returncode,
        "stdout": _decode_process_output(proc.stdout),
        "stderr": _decode_process_output(proc.stderr),
    }


def _read_doc_with_antiword(tmp_path: str) -> str:
    try:
        result = _run_external_tool(["antiword", "-w", "0", tmp_path])
    except Exception:
        return ""
    if result["returncode"] == 0 and result["stdout"]:
        return result["stdout"]
    return ""


def _read_doc_with_catdoc(tmp_path: str) -> str:
    try:
        result = _run_external_tool(["catdoc", "-d", "utf-8", tmp_path])
    except Exception:
        return ""
    if result["stdout"]:
        return result["stdout"]
    return ""


def _read_doc(b: bytes) -> str:
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tf:
            tf.write(b)
            tf.flush()
            tmp_path = tf.name

        text = _read_doc_with_antiword(tmp_path)
        if not text:
            text = _read_doc_with_catdoc(tmp_path)
        return (text or "").strip()
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def read_file_to_text(file_bytes: bytes, filename: Optional[str] = None) -> str:
    fname = (filename or "").lower()
    ext = Path(fname).suffix

    if ext == ".docx":
        return _read_docx(file_bytes)
    if ext == ".pdf":
        return _read_pdf(file_bytes)
    if ext == ".rtf":
        return _read_rtf(file_bytes)
    if ext == ".doc":
        return _read_doc(file_bytes)
    if ext == ".txt":
        return _decode_txt_best_effort(file_bytes).strip()

    if _is_pdf(file_bytes):
        return _read_pdf(file_bytes)
    if _docx_package_has_document_xml(file_bytes):
        return _read_docx(file_bytes)
    if _is_rtf(file_bytes):
        return _read_rtf(file_bytes)
    if _is_doc_ole(file_bytes):
        return _read_doc(file_bytes)

    return _decode_txt_best_effort(file_bytes).strip()


def read_bytes_to_text(file_bytes: bytes, filename: Optional[str] = None) -> str:
    return read_file_to_text(file_bytes, filename=filename)


def get_text_from_file(file_bytes: bytes, filename: Optional[str] = None) -> str:
    return read_file_to_text(file_bytes, filename=filename)
