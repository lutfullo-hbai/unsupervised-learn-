"""PDF matnini sahifa darajasida ajratib olish (I/O contract §1, §2.1).

Bu modul **pipeline'ning birinchi bosqichi** — undan keyingi barcha
qadamlar (M3 preprocessing, M4 chunking, M5 TF-IDF) shu matn ustida
ishlaydi. Shu sabab bu yerda aniqlanadigan xatolar butun pipeline'ni
to'xtatadi: yarim ish qilingan clustering'dan ko'ra aniq xato yaxshi.

Tekshiruvlar `docs/io_contract.md` §1.2 dagi ro'yxatga to'liq amal qiladi:

| # | Qoida | Natija |
|---|-------|--------|
| I1 | magic bytes `%PDF` | `InvalidPdfError` |
| I2 | fayl bo'sh emas | `InvalidPdfError` |
| I4 | jami belgi >= `min_total_chars` | `ScannedPdfError` |
| I5 | `needs_pass == False` | `InvalidPdfError` |
| I6 | `pymupdf.open()` muvaffaqiyatli | `InvalidPdfError` |
| I7 | kamida 1 sahifa | `InvalidPdfError` |

I3 (fayl hajmi) **bu modalda emas** — u API qatlamining mas'uliyati
(`MAX_UPLOAD_MB`), chunki bu modul CLI/notebook'dan ham chaqiriladi
va u fayl hajmini oldindan bilmaydi.

**Nima qasddan qilinmaydi:** OCR (non-goal), parol bilan ochish,
rasmdan matn topish, jadvalni tiklash.
"""

from pathlib import Path

import pymupdf

from app.core.exceptions import InvalidPdfError, ScannedPdfError
from app.schemas.document import PageText

__all__ = ["MAGIC_BYTES", "extract_pages", "has_pdf_magic"]

#: PDF formatining boshlanish belgisi — `docs/io_contract.md` §1.2 (I1).
#: PDF spetsifikatsiyasiga ko'ra `%PDF-` (5 bayt) boshlanadi. `==` emas,
#: `startswith` ishlatiladi: `.pdf` kengaytmasi boshqa formatlarda ham
#: uchraydi, lekin bu boshlanish **faqat** PDF'ga xos.
MAGIC_BYTES = b"%PDF"

#: `pymupdf.open()` qaytaradigan xatolar boshqa kutubxonaga tegishli.
#: Ular ro'yxati vaqt bilan o'zgaradi, shu sabab biz **barcha**
#: `Exception` ni ushlab, o'zimizning aniq xatomizga aylantiramiz.
_OPEN_ERROR_HINT = "PDF faylni ochib bo'lmadi"


def has_pdf_magic(head: bytes) -> bool:
    """Berilgan baytlarning boshida PDF belgisi bormi.

    Args:
        head: faylning **dastlabki baytlari** (odatda 5-8 bayt).

    Returns:
        `True` agar `%PDF` bilan boshlanadigan bo'lsa.

    >>> has_pdf_magic(b"%PDF-1.7 ...")
    True
    >>> has_pdf_magic(b"\\x89PNG\\r\\n")
    False
    """
    return head.startswith(MAGIC_BYTES)


def _read_head(source: str | Path | bytes, size: int) -> bytes:
    """Manbani o'zmagan holda dastlabki `size` baytni o'qiydi."""
    if isinstance(source, bytes):
        return source[:size]
    with open(source, "rb") as fh:
        return fh.read(size)


def extract_pages(
    source: str | Path | bytes,
    *,
    min_total_chars: int | None = None,
) -> list[PageText]:
    """PDF dan sahifa-darajasidagi matnni oladi.

    Args:
        source: PDF manbai — fayl yo'li (`str`/`Path`) yoki faylning
            o'zi (`bytes`, FastAPI `UploadFile` uchun).
        min_total_chars: Skanerlangan PDF chegarasi. `None` bo'lsa
            `Settings.min_total_chars` (`.env`) ishlatiladi.

    Returns:
        Sahifalar ro'yxati, **1-based** raqam bilan va kirish tartibida.
        Bo'sh sahifalar **saqlanadi** — ular keyingi bosqichlarda
        muhim (muqova, ajratgich).

    Raises:
        InvalidPdfError: fayl yo'q, bo'sh, `%PDF` emas, buzilgan,
            parol bilan himoyalangan yoki sahifasi yo'q.
        ScannedPdfError: ochildi, lekin jami matn chegaradan kam.

    Example:
        >>> pages = extract_pages("data/raw/01_short_ml_basics.pdf")
        >>> pages[0].page, pages[0].n_chars > 0
        (1, True)
    """
    if min_total_chars is None:
        from app.core.config import get_settings

        min_total_chars = get_settings().min_total_chars

    _validate_head(source)
    doc = _open(source)
    try:
        _validate_document(doc)
        return _read_pages(doc, min_total_chars)
    finally:
        # `pymupdf` ochiq fayl descriptor'ini ushlab turadi — Windows'da
        # faylni o'chirish yoki qayta yozish bloklanmasligi uchun
        # har doim yopamiz.
        doc.close()


def _validate_head(source: str | Path | bytes) -> None:
    """I1 va I2: fayl bo'sh emas va `%PDF` bilan boshlanadi."""
    try:
        head = _read_head(source, len(MAGIC_BYTES) + 1)
    except FileNotFoundError as exc:
        raise InvalidPdfError(f"Fayl topilmadi: {source}") from exc
    except IsADirectoryError as exc:
        raise InvalidPdfError(f"Bu papka, fayl emas: {source}") from exc
    except OSError as exc:
        raise InvalidPdfError(f"Faylni o'qib bo'lmadi: {source} ({exc})") from exc

    if not head:
        raise InvalidPdfError(f"{source}: fayl bo'sh (0 bayt).")

    if not has_pdf_magic(head):
        preview = head[:8]
        raise InvalidPdfError(
            f"{source}: PDF formatida emas — boshlanish {preview!r}, "
            f"kutilgan {MAGIC_BYTES!r}. Kengaytma `.pdf` yetarli "
            f"ishonchli signal emas."
        )


def _open(source: str | Path | bytes) -> pymupdf.Document:
    """I6: PDF'ni ochadi, muvaffaqiyatsizlikni `InvalidPdfError` ga aylantiradi."""
    try:
        if isinstance(source, bytes):
            return pymupdf.open(stream=source, filetype="pdf")
        return pymupdf.open(source)
    except Exception as exc:
        # `pymupdf` xatolari keng spektrli (FileDataError, EmptyFileError,
        # RuntimeError, ValueError ...) va ro'yxati versiyaga bog'liq.
        # Bizminga faqat bitta nazorat qiziq: "bu PDF emas".
        raise InvalidPdfError(f"{_OPEN_ERROR_HINT}: {source} ({type(exc).__name__}: {exc})") from exc


def _validate_document(doc: pymupdf.Document) -> None:
    """I5 va I7: parol yo'q va kamida bitta sahifa bor."""
    if doc.needs_pass:
        raise InvalidPdfError(
            "PDF parol bilan himoyalangan. Parol bilan ochish qo'llab-quvvatlanmaydi "
            "(docs/io_contract.md §1.3)."
        )

    n_pages = len(doc)
    if n_pages < 1:
        raise InvalidPdfError(f"PDF'da sahifa yo'q (topildi: {n_pages}).")


def _read_pages(doc: pymupdf.Document, min_total_chars: int) -> list[PageText]:
    """I4: sahifalarni o'qiydi va matn chegarasini tekshiradi."""
    pages: list[PageText] = []
    for number, page in enumerate(doc, start=1):
        # `sort=True` **ishlatilmaydi** — bu noto'g'ri taxmin edi
        # (docs/limitations.md L-01). `sort=True` matnni vertikal
        # koordinat bo'yicha saralaydi va ikki ustunli hujjatda
        # chap/o'ng ustunni qatorlab aralashib ketiradi. Standart
        # (`sort=False`) esa content stream tartibini saqlaydi — bu
        # ko'pchilik real hujjatlarda o'qish tartibi bilan bir xil.
        text = page.get_text("text").strip()
        pages.append(PageText(page=number, text=text))

    total = sum(page.n_chars for page in pages)
    if total < min_total_chars:
        raise ScannedPdfError(
            f"Matn qatlami deyarli yo'q: {total} belgi < {min_total_chars}. "
            f"Bu skanerlangan PDF bo'lishi mumkin — OCR qo'llab-quvvatlanmaydi "
            f"(docs/io_contract.md §1.3)."
        )

    return pages