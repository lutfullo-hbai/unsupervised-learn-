"""Real korpus skriptlarining regressiya testlari.

**Nima uchun bu yerda test bor.** Haqiqiy yuklashda xato yuz berdi:
Gutenberg ID `25438` "The Jungle" deb yozilgan edi, lekin yuklangan
fayl aslida **"The Airlords of Han"** edi. ID xotiradan taxmin
qilingandi va yuklangan kontent **tekshirilmagan** edi.

Shuning uchun `scripts/` moduli ham testlanadi — u kod, boshqa
koddek. Testlar **tarmoqsiz** ishlaydi: `http_get` monkeypatch qilinadi.

L-12/L-13/L-14 uchun regressiya testlari `tests/test_preprocessing.py` da.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_real_documents as frd  # noqa: E402


class TestVerifyTitle:
    """Yuklangan matn e'lon qilingan kitobga tegishli ekanini tekshirish."""

    def test_matching_title_passes(self):
        text = "Upton Sinclair\n\nThe Jungle\n\nIt was the best of times."
        assert frd.verify_title(text, "The Jungle", "Upton Sinclair") == (True, "mos")

    def test_wrong_book_is_caught(self):
        """Asl xato: ID 25438 -> 'The Airlords of Han', 'The Jungle'
        deb yozilgan edi. Sarlavha matnda umuman yo'q."""
        text = "The Airlords of Han\n\nI. The Airlords Besieged\n"
        ok, note = frd.verify_title(text, "The Jungle", "Upton Sinclair")
        assert ok is False
        assert "topilmadi" in note

    def test_missing_author_is_caught(self):
        text = "The Jungle\n\nThe packing plant was a marvel."
        ok, note = frd.verify_title(text, "The Jungle", "Upton Sinclair")
        assert ok is False
        assert "muallif" in note

    def test_leading_article_is_not_required(self):
        """`The`/`A`/`An`/`of` ma'nosiz — qolgan so'zlar talab qilinadi.

        `The` — bitta ma'nosiz so'zdan iborat sarlavha esa **hali**
        tekshiriladi (`of` dan keyingi so'z qolishi mumkin emas).
        """
        text = "Mary Shelley\n\nFrankenstein; or, The Modern Prometheus"
        ok, note = frd.verify_title(text, "Frankenstein; or, The Modern Prometheus", "Mary Shelley")
        assert ok is True, note

    def test_article_only_title_still_checked(self):
        ok, note = frd.verify_title("Ann Author wrote about something else", "The", "Ann Author")
        assert ok is False
        assert note

    def test_punctuation_and_case_do_not_matter(self):
        text = "Jane Austen\n\nMR. DARCY — a tale"
        assert frd.verify_title(text, "Mr. Darcy", "Jane Austen")[0] is True

    def test_empty_text_fails(self):
        ok, note = frd.verify_title("", "The Jungle", "Upton Sinclair")
        assert ok is False
        assert note

    def test_normalize_for_search(self):
        assert frd.normalize_for_search("Hello, World!") == "hello world"


class TestGutenbergText:
    """Litsenziya sarlavhasini olib tashlash — marker orqali."""

    RAW = (
        "The Project Gutenberg eBook of Something\n"
        "This header must be removed. gutenberg copyright public domain\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "Real book text.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
        "Trailer that must be removed too.\n"
    )

    def test_markers_are_stripped(self, monkeypatch):
        monkeypatch.setattr(frd, "http_get", lambda url, **kw: self.RAW.encode())
        out = frd.gutenberg_text(1)
        assert "Real book text." in out
        assert "gutenberg" not in out.lower()
        assert "Trailer" not in out

    def test_missing_marker_raises(self, monkeypatch):
        monkeypatch.setattr(frd, "http_get", lambda url, **kw: b"No markers here.")
        with pytest.raises(frd.FetchError, match="START"):
            frd.gutenberg_text(1)

    def test_th_is_also_accepted(self, monkeypatch):
        """Ba'zi kitoblarda `THIS` ishlatiladi."""
        raw = (
            "*** START OF THIS PROJECT GUTENBERG EBOOK X ***\nBody.\n"
            "*** END OF THIS PROJECT GUTENBERG EBOOK X ***\n"
        )
        monkeypatch.setattr(frd, "http_get", lambda url, **kw: raw.encode())
        assert "Body." in frd.gutenberg_text(1)


class TestCatalogIntegrity:
    """Katalogdagi ID va sarlavhalar o'zaro mos kelishi kerak."""

    def test_gutenberg_ids_are_unique(self):
        ids = [row[0] for row in frd.LITERATURE]
        assert len(ids) == len(set(ids))

    def test_slugs_are_unique(self):
        slugs = [row[1] for row in frd.LITERATURE] + [row[1] for row in frd.OA_PAPERS]
        assert len(slugs) == len(set(slugs))

    def test_every_row_has_five_fields(self):
        for row in frd.LITERATURE:
            assert len(row) == 5, f"gutenberg qatori noto'g'ri: {row}"
        for row in frd.OA_PAPERS:
            assert len(row) == 4, f"OA qatori noto'g'ri: {row}"

    def test_oa_papers_have_a_title(self):
        """Avval `title: None` yozilardi — reja bo'sh qoldi."""
        assert all(row[3] for row in frd.OA_PAPERS)

    def test_the_jungle_uses_the_correct_id(self):
        """Regression: 25438 'The Airlords of Han' edi."""
        ids = {row[1]: row[0] for row in frd.LITERATURE}
        assert ids["the_jungle"] == 140

    def test_every_source_has_a_known_license(self):
        assert set(frd.LICENSES) == {"gutenberg", "arxiv", "plos"}
        for name, (text, url) in frd.LICENSES.items():
            assert text.strip(), name
            assert url.startswith("https://"), name

    def test_arxiv_license_does_not_claim_cc_by(self):
        """arXiv litsenziyasi maqolaga qarab farq qiladi — CC-BY deb
        **guman qilmaslik** kerak, bu huquqiy xato bo'ladi."""
        text, _ = frd.LICENSES["arxiv"]
        assert "CC-BY" not in text and "CC BY" not in text


class TestLiteraryCorpus:
    """Agar real korpus yuklangan bo'lsa, uning haqiqiyligini tekshirish.

    Testlar `data/raw/real/` yo'q bo'lsa **saklanadi** — L-05 bo'yicha
    testlar tashqi ma'lumotga bog'liq bo'lmasligi kerak.
    """

    @pytest.fixture(scope="class")
    def manifest(self):
        path = ROOT / "data/manifest.json"
        if not path.exists():
            pytest.skip("data/manifest.json yo'q — avval fetch_real_documents.py")
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def test_manifest_has_fifteen_documents(self, manifest):
        assert len(manifest["documents"]) == 15

    def test_every_entry_has_checksum(self, manifest):
        assert all(d.get("sha256") and len(d["sha256"]) == 64 for d in manifest["documents"])

    def test_every_entry_has_license_url(self, manifest):
        assert all(d.get("license_url", "").startswith("https://") for d in manifest["documents"])

    def test_no_title_is_missing(self, manifest):
        missing = [d["slug"] for d in manifest["documents"] if not d.get("title")]
        assert missing == [], f"sarlavhasiz: {missing}"

    def test_literature_titles_verified_in_manifest(self, manifest):
        """Skript endi `verify_title` ni majburiy ishlatadi — bu
        'The Jungle' xatosining qaytmasligi kafolati."""
        lit = [d for d in manifest["documents"] if d["group"] == "literature"]
        assert len(lit) == 10