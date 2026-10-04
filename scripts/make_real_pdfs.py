"""Gutenberg adabiyotini PDF'ga o'girish — pipeline shuni iste'mol qiladi.

**Nima uchun.** `app.services.extract_pages` **PDF** oladi. Project
Gutenberg esa faqat matn beradi. Adabiyot matnini PDF'ga o'girish
kerak, aks holda real korpus ishlatilmay qoladi.

**Bu PDF'lar qadar "haqiqiy"?** Ochiq aytish kerak:

| Xususiyat | Manba | Adabiyot PDF'i |
|-----------|-------|----------------|
| Matn mazmuni | ✅ haqiqiy kitob | ✅ **haqiqiy** |
| Justified ( tenglashgan) satr | ✅ | ✅ pymupdf `Story` |
| Sahifa raqamlari | ✅ | ✅ shu skript qo'yadi |
| Defis bilan uzilgan so'z | ✅ haqiqiy kitoblarda bor | ❌ **yo'q** |
| Running header | ✅ | ❌ **ataylab qo'yilmadi** |
| Ligatura | ✅ ba'zi kitoblarda | ❌ **yo'q** |

Ya'ni adabiyot PDF'lari **mazmun** jihatidan haqiqiy, **artefakt**
jihatidan esa sun'iy. Artefaktlarni haqiqiy olish uchun
`data/raw/real/pdf/` dagi arXiv/PLOS fayllari ishlatiladi — ular
tashqi manbadan olingan, tegilmagan.

**Nega header ataylab qo'yilmadi:** sun'iy yaratilgan header'ni
`remove_repeated_headers_footers` ga berib, "haqiqiy header tozalash
ishlaydi" deb **yolg'on** xulosa chiqarish mumkin. Artefakt dalili
faqat tashqi PDF'lardan olinadi.

Ishga tushirish:
    .venv/bin/python scripts/make_real_pdfs.py
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import pymupdf

TEXT_DIR = Path("data/raw/real/gutenberg_text")
PDF_DIR = Path("data/raw/real/pdf")
MANIFEST = Path("data/manifest.json")

#: Kitob o'lchami — A5 (148x210 mm), chekkalar 20 mm.
PAGE_W, PAGE_H = 420.0, 595.0
MARGIN = 20.0

_FONT_CSS = (
    "font-family:serif; font-size:10.5pt; line-height:1.45; text-align:justify"
)
_HEAD_CSS = "font-family:serif; font-size:12.5pt; text-align:center; font-weight:bold"


def looks_like_heading(block: str) -> bool:
    """Blok sarlavha ekanini aniqlaydi.

    Gutenberg matnida sarlavhalar qisqa, ko'pincha katta harfli
    (`CHAPTER I.`) yoki `CONTENTS`/`PREFACE` ko'rinishida bo'ladi.

    Args:
        block: bo'sh qatorlardan ajratilgan bir blok.

    Returns:
        Sarlavha deb hisoblanganda `True`.
    """
    text = block.strip()
    if not text or len(text) > 70 or "\n" in text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    if sum(c.isupper() for c in letters) / len(letters) > 0.8:
        return True
    return bool(re.match(r"^(chapter|book|part|act|scene|volume)\b", text, re.I))


def text_to_html(text: str, *, title: str) -> str:
    """Gutenberg matnini justified HTML'ga o'giradi.

    Args:
        text: toza kitob matni.
        title: `<title>` uchun.

    Returns:
        pymupdf `Story` uchun HTML.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    parts: list[str] = [
        "<html><head>"
        f"<style>p{{{_FONT_CSS}}} h1{{{_HEAD_CSS}}} h2{{{_HEAD_CSS}}}</style>"
        f"<title>{html.escape(title)}</title></head><body>"
    ]
    for block in blocks:
        safe = html.escape(block).replace("\n", " ")
        tag = "h1" if looks_like_heading(block) else "p"
        parts.append(f"<{tag}>{safe}</{tag}>")
    parts.append("</body></html>")
    return "".join(parts)


def render_pdf(html_text: str, dest: Path) -> int:
    """HTML'ni PDF'ga chiqadi va sahifa raqamlarini qo'yadi.

    Sahifa raqamlari **real** — paginatsiyalangan har bir kitobda
    bor. Lekin bu biz qo'shgan narsa: shuning uchun manifestda
    `page_numbers_added_by_us` deb belgilangan.

    Args:
        html_text: render qilinadigan HTML.
        dest: yoziladigan PDF.

    Returns:
        Sahifalar soni.
    """
    if dest.exists():
        dest.unlink()

    mediabox = pymupdf.Rect(0, 0, PAGE_W, PAGE_H)
    where = mediabox + (MARGIN, MARGIN + 12, -MARGIN, -MARGIN)

    story = pymupdf.Story(html=html_text)
    writer = pymupdf.DocumentWriter(str(dest))
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    # Sahifa raqamlari — markazda, pastda.
    with pymupdf.open(dest) as doc:
        for number, page in enumerate(doc, start=1):
            page.insert_text(
                (PAGE_W / 2 - 10, PAGE_H - 10),
                str(number),
                fontname="helv",
                fontsize=8,
                color=(0, 0, 0),
            )
        pages = doc.page_count
        doc.saveIncr()
    return pages


def main() -> int:
    """Barcha `.txt` fayllarni PDF'ga o'giradi va manifest'ni yangilaydi.

    Returns:
        `0` — hammasi muvaffaqiyatli, `1` — kamida bitta xato.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="faqat birinchi N ta kitob")
    args = parser.parse_args()

    sources = sorted(TEXT_DIR.glob("*.txt"))
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        print(f"{TEXT_DIR} bo'sh. Avval: .venv/bin/python scripts/fetch_real_documents.py")
        return 1

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    by_slug = {d["slug"]: d for d in manifest.get("documents", [])}

    failures = 0
    print("=== Adabiyot -> PDF ===")
    for src in sources:
        slug = src.stem
        dest = PDF_DIR / f"lit_{slug}.pdf"
        meta = by_slug.get(slug, {})
        try:
            pages = render_pdf(
                text_to_html(src.read_text(encoding="utf-8"), title=meta.get("title", slug)),
                dest,
            )
            with pymupdf.open(dest) as doc:
                chars = sum(len(p.get_text()) for p in doc)
            if meta:
                meta.update(
                    {
                        "file": str(dest),
                        "format": "pdf",
                        "bytes": dest.stat().st_size,
                        "pages": pages,
                        "chars": chars,
                        "derived_from": str(src),
                        "page_numbers_added_by_us": True,
                        "headers_added_by_us": False,
                    }
                )
            print(f"  OK  lit_{slug:26} {pages:>4} sahifa  {chars:>9,} belgi")
        except Exception as exc:  # noqa: BLE001 — skript bitta kitobdan to'xtamasin
            failures += 1
            print(f"  XATO lit_{slug:26} {type(exc).__name__}: {exc}")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest yangilandi: {MANIFEST} ({failures} ta xato)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())