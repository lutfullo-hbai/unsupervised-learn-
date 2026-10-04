"""M2: PDF matnini ajratib olish testlari.

Test nomlari **xulqni** tushuntiradi (qa §63):
`extract_pages_raises_invalid_pdf_when_file_has_no_pdf_magic`.

Muhim: testlar `data/raw/` ga **bog'liq emas** — u `.gitignore`da
(katta PDF'lar git'ga kirmaydi), shuning uchun klonlangan repo'da
bu fayllar bo'lmaydi. Barcha PDF'lar test ichida `pymupdf` bilan
yaratiladi — testlar o'z-o'zidan yetarli.
"""

import os
from pathlib import Path

import pymupdf
import pytest

from app.core.exceptions import (
    InvalidPdfError,
    NotEnoughContentError,
    PdfAnalyzerError,
    ScannedPdfError,
)
from app.core.config import Settings
from app.schemas import PageText
from app.services import extract_pages
from app.services.pdf_extractor import has_pdf_magic

# --- Yordamchilar ------------------------------------------------------


def make_text_pdf(path: Path, pages: list[str]) -> Path:
    """Har biri `pages[i]` matnidan iborat bo'limli PDF yaratadi."""
    doc = pymupdf.open()
    for content in pages:
        page = doc.new_page()
        # `insert_textbox` matnni rect ichiga **o'raydi**; `insert_text`
        # esa sahifa chegarasidan tashqariga chiqsa matnni kesib qo'yadi
        # (keyin `get_text` qisqarilgan matnni qaytaradi).
        # "helv" (Helvetica) lotin belgilarini to'liq qo'llaydi — bu
        # testlarda ASCII matn yetarli, shrift tanlash muammosi bo'lmaydi.
        page.insert_textbox(
            pymupdf.Rect(72, 72, 523, 770),
            content,
            fontname="helv",
            fontsize=12,
        )
    doc.save(path)
    doc.close()
    return path


def make_scanned_pdf(path: Path, n_pages: int = 1) -> Path:
    """Matn qatlami **yo'q** PDF — skanerlangan hujjatni taqlid qiladi."""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200))
    pix.set_rect(pix.irect, (128, 128, 128))
    for _ in range(n_pages):
        doc.new_page().insert_image(pymupdf.Rect(0, 0, 200, 200), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def make_encrypted_pdf(src: Path, dest: Path, user_pw: str = "secret") -> Path:
    """Parol bilan himoyalangan PDF."""
    with pymupdf.open(src) as doc:
        dest.write_bytes(
            doc.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="owner",
                user_pw=user_pw,
            )
        )
    return dest


def make_png_bytes() -> bytes:
    """`.pdf` nomi berilishi mumkin bo'lgan PNG — magic bytes ni sinab ko'rish uchun."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
    return pix.tobytes("png")


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """3 sahifali, matnli, chegaradan oshgan PDF."""
    return make_text_pdf(
        tmp_path / "report.pdf",
        [
            "Supervised learning trains a model on labelled examples. " * 2,
            "Unsupervised learning finds structure without labels. " * 2,
            "Evaluation measures how well a model generalises. " * 2,
        ],
    )


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    return make_scanned_pdf(tmp_path / "scanned.pdf", n_pages=2)


# --- M2.T2: sxema va xatolar ------------------------------------------


def test_page_text_rejects_zero_page_number():
    """Sahifa raqami 1-based — 0 foydalanuvchi ko'rmaydi."""
    with pytest.raises(ValueError):
        PageText(page=0, text="salom")


def test_page_text_reports_char_count_and_emptiness():
    page = PageText(page=1, text="salom")
    assert page.n_chars == 5
    assert page.is_empty is False
    assert PageText(page=2, text="   ").is_empty is True


def test_all_domain_errors_inherit_from_pdf_analyzer_error():
    for error in (InvalidPdfError, ScannedPdfError, NotEnoughContentError):
        assert issubclass(error, PdfAnalyzerError)


def test_scanned_and_invalid_errors_are_distinguishable():
    """API qatlami (M13) shu farq bilan 400 va 422 ajratadi."""
    assert not issubclass(ScannedPdfError, InvalidPdfError)
    assert not issubclass(InvalidPdfError, ScannedPdfError)


# --- M2.T3: magic bytes ------------------------------------------------


@pytest.mark.parametrize(
    "head",
    [b"%PDF", b"%PDF-1.7", b"%PDF-2.0", b"%PDF-1.4\n%\xe2\xe3"],
)
def test_has_pdf_magic_accepts_pdf_headers(head):
    assert has_pdf_magic(head) is True


@pytest.mark.parametrize(
    "head",
    [b"", b"%PD", b"%pd-1.7", b"\x89PNG\r\n", b"PK\x03\x04", b" hello"],
)
def test_has_pdf_magic_rejects_non_pdf_headers(head):
    assert has_pdf_magic(head) is False


def test_has_pdf_magic_is_case_sensitive():
    """`%pdf` — PDF formatida emas; kichik harfni qabul qilish xato bo'lardi."""
    assert has_pdf_magic(b"%pdf-1.7") is False


# --- M2.T3: muvaffaqiyatli o'qish --------------------------------------


def test_extract_pages_returns_pages_numbered_from_one(text_pdf: Path):
    pages = extract_pages(text_pdf)
    assert [p.page for p in pages] == [1, 2, 3]


def test_extract_pages_returns_text_for_every_page(text_pdf: Path):
    pages = extract_pages(text_pdf)
    assert all(p.n_chars > 0 for p in pages)
    assert "Supervised learning" in pages[0].text
    assert "Unsupervised learning" in pages[1].text


def test_extract_pages_strips_whitespace(text_pdf: Path):
    """`text` `.strip()` qilingan bo'lishi shart (io_contract §2.1)."""
    for page in extract_pages(text_pdf):
        assert page.text == page.text.strip()


def test_extract_pages_accepts_path_string_and_path_object_identically(text_pdf: Path):
    from_str = extract_pages(str(text_pdf))
    from_path = extract_pages(Path(text_pdf))
    assert [p.text for p in from_str] == [p.text for p in from_path]


def test_extract_pages_accepts_bytes_like_fastapi_upload(text_pdf: Path):
    """FastAPI `UploadFile` bytes beradi — natija path bilan bir xil bo'lishi shart."""
    raw = text_pdf.read_bytes()
    assert [p.text for p in extract_pages(raw)] == [p.text for p in extract_pages(text_pdf)]


def test_extract_pages_preserves_page_order(text_pdf: Path):
    """Sahifa tartibi buzilsa, M12 da manba havolalari chalkashadi."""
    pages = extract_pages(text_pdf)
    assert "Supervised" in pages[0].text
    assert "Evaluation" in pages[2].text


def test_extract_pages_keeps_blank_pages_instead_of_dropping_them(tmp_path: Path):
    """Bo'sh sahifalar statistika uchun saqlanadi (io_contract §2.1)."""
    pdf = make_text_pdf(
        tmp_path / "mixed.pdf",
        ["", "Real content on the second page. " * 10, ""],
    )
    pages = extract_pages(pdf)
    assert [p.page for p in pages] == [1, 2, 3]
    assert pages[0].is_empty and pages[2].is_empty
    assert not pages[1].is_empty


def test_extract_pages_is_deterministic_across_calls(text_pdf: Path):
    """`sort=True` barcha sahifalarda bir xil natija berishi shart."""
    assert [p.text for p in extract_pages(text_pdf)] == [
        p.text for p in extract_pages(text_pdf)
    ]


def test_extract_pages_preserves_column_reading_order(tmp_path: Path):
    """L-01: `sort=True` ishlatilmasligi regressiya testi.

    Ikki ustunli hujjatda PyMuPDF ning `sort=True` varianti chap va o'ng
    ustunni qatorlab aralashib ketiradi. Standart (`sort=False`) esa
    o'qish tartibini saqlaydi. Bu test yorliq yo'qligini kafolatlaydi.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 60, 290, 760), "LEFT-COLUMN-LINE\n" * 8, fontname="helv", fontsize=9
    )
    page.insert_textbox(
        pymupdf.Rect(310, 60, 550, 760), "RIGHT-COLUMN-LINE\n" * 8, fontname="helv", fontsize=9
    )
    path = tmp_path / "two_col.pdf"
    doc.save(path)
    doc.close()

    text = extract_pages(path)[0].text

    assert text.index("LEFT-COLUMN-LINE") < text.index("RIGHT-COLUMN-LINE"), (
        "chap ustun o'ng ustundan keyin keldi — ustunlar aralashgan"
    )
    # `sort=True` ishlatilgan bo'lsa, birinchi RIGHT dan keyin yana LEFT
    # qaytib kelardi (chiziq-ma-chiziq aralashuv).
    assert text.rfind("LEFT-COLUMN-LINE") < text.index("RIGHT-COLUMN-LINE"), (
        "o'ng ustundan keyin yana chap ustun keldi — qatorlab aralashgan"
    )


def test_extract_pages_reads_min_total_chars_from_settings(tmp_path: Path):
    """Chegara kodga yozilmaydi — `.env` dan o'qiladi (qa §18)."""
    assert Settings().min_total_chars == 100


def test_extract_pages_honours_explicit_min_total_chars_overrides(text_pdf: Path):
    assert len(extract_pages(text_pdf, min_total_chars=0)) == 3


# --- M2.T3: xatolar (I1, I2, I5, I6, I7) ------------------------------


def test_extract_pages_raises_invalid_pdf_when_file_does_not_exist(tmp_path: Path):
    with pytest.raises(InvalidPdfError, match="topilmadi"):
        extract_pages(tmp_path / "yoq.pdf")


def test_extract_pages_raises_invalid_pdf_when_source_is_a_directory(tmp_path: Path):
    with pytest.raises(InvalidPdfError, match="papka"):
        extract_pages(tmp_path)


def test_extract_pages_raises_invalid_pdf_when_file_is_empty(tmp_path: Path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(InvalidPdfError, match="bo'sh"):
        extract_pages(empty)


def test_extract_pages_raises_invalid_pdf_when_file_has_no_pdf_magic(tmp_path: Path):
    """I1 — `.pdf` kengaytmasi yetarli emas (io_contract §1.2)."""
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(make_png_bytes())
    with pytest.raises(InvalidPdfError, match="PDF formatida emas"):
        extract_pages(fake)


def test_extract_pages_rejects_plain_text_file_with_pdf_extension(tmp_path: Path):
    fake = tmp_path / "notes.pdf"
    fake.write_text("bu oddiy matn, PDF emas", encoding="utf-8")
    with pytest.raises(InvalidPdfError, match="PDF formatida emas"):
        extract_pages(fake)


def test_extract_pages_raises_invalid_pdf_when_document_is_corrupted(tmp_path: Path, text_pdf: Path):
    """I6 — header butun, jism buzilgan: faqat `pymupdf.open()` ushlaydi."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(text_pdf.read_bytes()[:400])
    with pytest.raises(InvalidPdfError, match="ochib bo'lmadi"):
        extract_pages(broken)


def test_extract_pages_raises_invalid_pdf_when_pdf_is_password_protected(
    tmp_path: Path, text_pdf: Path
):
    """I5 — parol bilan PDF ochiladi, lekin matn bo'sh qaytaradi.

    Shu sabab `needs_pass` alohida tekshiriladi: aks holda bu holat
    "skanerlangan" deb xato baholanadi.
    """
    encrypted = make_encrypted_pdf(text_pdf, tmp_path / "enc.pdf")
    with pytest.raises(InvalidPdfError, match="parol"):
        extract_pages(encrypted)


def test_extract_pages_raises_scanned_pdf_when_text_layer_is_missing(scanned_pdf: Path):
    """I4 — ochildi, lekin matn qatlami yo'q. OCR non-goal."""
    with pytest.raises(ScannedPdfError, match="Matn qatlami"):
        extract_pages(scanned_pdf)


def test_scanned_error_message_explains_ocr_is_unavailable(scanned_pdf: Path):
    with pytest.raises(ScannedPdfError, match="OCR"):
        extract_pages(scanned_pdf)


def test_scanned_pdf_is_rejected_for_bytes_source_too(scanned_pdf: Path):
    with pytest.raises(ScannedPdfError):
        extract_pages(scanned_pdf.read_bytes())


def test_extract_pages_accepts_scanned_pdf_when_threshold_is_zero(scanned_pdf: Path):
    """Chegara 0 bo'lsa skanerlangan PDF o'tadi — chegara haqiqatan qo'llaniladi."""
    pages = extract_pages(scanned_pdf, min_total_chars=0)
    assert len(pages) == 2
    assert all(p.is_empty for p in pages)


def test_short_but_readable_pdf_is_rejected_under_default_threshold(tmp_path: Path):
    """Chiqarish chegarasidan (100) kam matn — `ScannedPdfError`.

    chegarasiz qabul qilinsa, pipeline keyin chalkash bo'sh natija
    berardi (`docs/io_contract.md` §4).
    """
    short = make_text_pdf(tmp_path / "tiny.pdf", ["Qisqa matn."])
    with pytest.raises(ScannedPdfError):
        extract_pages(short)


def test_extract_pages_raises_for_empty_bytes_input():
    with pytest.raises(InvalidPdfError, match="bo'sh"):
        extract_pages(b"")


def test_extract_pages_raises_for_non_pdf_bytes_input():
    with pytest.raises(InvalidPdfError, match="PDF formatida emas"):
        extract_pages(make_png_bytes())


def test_extract_pages_rejects_truncated_bytes_input(text_pdf: Path):
    with pytest.raises(InvalidPdfError):
        extract_pages(text_pdf.read_bytes()[:400])


def test_extract_pages_raises_invalid_pdf_when_document_has_no_pages(tmp_path: Path):
    """I7 — `pymupdf` sahifasiz PDF'ni ochadi (bo'sh page tree)."""
    empty_tree = tmp_path / "no_pages.pdf"
    empty_tree.write_bytes(
        b"%PDF-1.7\n"
        b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R >>\n%%EOF"
    )
    with pymupdf.open(empty_tree) as doc:
        assert len(doc) == 0, "fixture haqiqatan sahifasiz bo'lishi kerak"

    with pytest.raises(InvalidPdfError, match="sahifa yo'q"):
        extract_pages(empty_tree)


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root barcha faylni o'qiydi — ruxsat rad etilishini tekshira olmaydi",
)
def test_extract_pages_raises_invalid_pdf_when_file_is_not_readable(tmp_path: Path, text_pdf: Path):
    """Ruxsat rad etilishi `FileNotFoundError` emas — umumiy `OSError` beradi."""
    locked = tmp_path / "locked.pdf"
    locked.write_bytes(text_pdf.read_bytes())
    locked.chmod(0o000)
    try:
        with pytest.raises(InvalidPdfError, match="o'qib bo'lmadi"):
            extract_pages(locked)
    finally:
        locked.chmod(0o644)


def test_extract_pages_closes_document_even_when_validation_fails(
    monkeypatch, scanned_pdf: Path
):
    """Xatoda ham fayl yopilishi kerak — aks holda descriptor oqib ketadi."""
    opened: list[pymupdf.Document] = []
    original_open = pymupdf.open

    def spy(*args, **kwargs):
        doc = original_open(*args, **kwargs)
        opened.append(doc)
        return doc

    monkeypatch.setattr(pymupdf, "open", spy)
    with pytest.raises(ScannedPdfError):
        extract_pages(scanned_pdf)

    assert opened, "pymupdf.open chaqirilmadi"
    assert all(doc.is_closed for doc in opened), "xatodan keyin hujjat yopilmadi"


def test_extract_pages_error_message_names_the_offending_file(tmp_path: Path):
    """Xato matni fayl nomini ko'rsatishi kerak — logda izlash oson bo'lsin."""
    broken = tmp_path / "mening_buzilgan_hujjatim.pdf"
    broken.write_bytes(b"not a pdf at all")
    with pytest.raises(InvalidPdfError) as exc:
        extract_pages(broken)
    assert "mening_buzilgan_hujjatim.pdf" in str(exc.value)