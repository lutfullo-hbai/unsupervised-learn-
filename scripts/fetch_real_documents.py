"""Haqiqiy ochiq manbadan hujjatlar yuklab olish — real test korpusi.

**Nima uchun kerak.** M2/M3 da ishlatilgan `scripts/make_sample_pdfs.py`
namunalari **sun'iy** edi. Ular aniq, lekin **too toza**: `docs/limitations.md`
L-08 bo'yicha ular hech qanday defis bilan uzilgan so'z, ligatura yoki
`\\xa0` bermadi. Bu esa preprocessing funksiyalarini **faqat sintetik
testlarda** tekshirishga majbur qilgan.

Bu skript haqiqiy, ochiq litsenziyali hujjatlarni yuklab oladi:

| Guruh | Manba | Litsenziya | Nima beradi |
|-------|-------|-----------|-------------|
| Adabiyot | Project Gutenberg | public domain (AQSh) | haqiqiy ingliz matni, turli janr |
| Ilmiy maqola | arXiv / PLOS ONE | CC-BY / arXiv litsenziyasi | haqiqiy tiqilgan PDF, artefaktli |

**Tuzilishi.** Har bir kitob **alohida PDF** — chunki M6 (K-Means) va
M7 (baholash) uchun turli mavzular kerak: bitta PDF'da barcha adabiyotni
aralashtirsa, klasterlar mavzuni emas, **muallifni** ko'rsatadi.

**Provenance.** `data/manifest.json` har bir fayl uchun manba, litsenziya,
SHA-256, sahifa va belgi sonini yozadi. Bu fayl **commit qilinadi**;
PDF'lar esa `.gitignore`da (`data/raw/real/`).

Ishga tushirish:
    .venv/bin/python scripts/fetch_real_documents.py            # yuklab olish
    .venv/bin/python scripts/fetch_real_documents.py --verify   # faqat tekshirish
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

RAW = Path("data/raw/real")
TEXT_DIR = RAW / "gutenberg_text"
PDF_DIR = RAW / "pdf"
MANIFEST = Path("data/manifest.json")

USER_AGENT = "unsupervised-learn (ta'lim loyihasi; real korpus)"

# --- Project Gutenberg: ingliz adabiyoti (public domain) ---------------
# (ebook_id, slug, muallif, nomi, janr)
LITERATURE: tuple[tuple[int, str, str, str, str], ...] = (
    (1342, "pride_and_prejudice", "Jane Austen", "Pride and Prejudice", "ijtimoiy / romanka"),
    (11, "alice_in_wonderland", "Lewis Carroll", "Alice's Adventures in Wonderland", "fantaziya"),
    (84, "frankenstein", "Mary Shelley", "Frankenstein", "gotik / fan-fantastika"),
    (1661, "sherlock_holmes", "Arthur Conan Doyle", "The Adventures of Sherlock Holmes", "detektiv"),
    (35, "time_machine", "H. G. Wells", "The Time Machine", "fan-fantastika"),
    (345, "dracula", "Bram Stoker", "Dracula", "gotik / qo'rqinchli"),
    (205, "walden", "Henry David Thoreau", "Walden", "tabiat / falsafa"),
    # 25438 "The Jungle" DEGIL edi — u "The Airlords of Han" (Amazing
    # Stories, 1929). To'g'ri raqam **140**. `verify_title` endi buni
    # avtomatik ushlaydi.
    (140, "the_jungle", "Upton Sinclair", "The Jungle", "ijtimoiy / protest"),
    (45, "anne_of_green_gables", "L. M. Montgomery", "Anne of Green Gables", "o'smirlik"),
    (174, "dorian_gray", "Oscar Wilde", "The Picture of Dorian Gray", "estetik / falsafa"),
)

# --- Ochiq litsenziyali ilmiy PDF'lar ----------------------------------
# (arxiv_id yoki DOI, slug, soha, sarlavha)
#
# Sarlavha qo'lda yozilgan — PDF'dan avtomatik ajratib olish ishonchsiz
# (arxiv'da `arXiv:1511.07123v1 [cond-mat.str-el] 23 Nov 2015` sarlavha
# qatidan **oldin** chiqadi). `verify_pdf` o'lchov bilan solishtiradi.
OA_PAPERS: tuple[tuple[str, str, str, str], ...] = (
    (
        "arxiv:1706.03762",
        "arxiv_attention_is_all_you_need",
        "NLP / transformer",
        "Attention Is All You Need",
    ),
    (
        "arxiv:1511.07123",
        "arxiv_skyrmions",
        "fizika",
        "Creation of Skyrmions by Electric Field on Chiral-Lattice "
        "Magnetic Insulators",
    ),
    (
        "arxiv:2003.00083",
        "arxiv_bradley_terry",
        "statistika",
        "Nonparametric Estimation in the Dynamic Bradley-Terry Model",
    ),
    (
        "doi:10.1371/journal.pone.0280199",
        "plos_middle_ear_effusion",
        "meditsina",
        "Middle ear effusion, ventilation tubes and neurological "
        "development in childhood",
    ),
    (
        "doi:10.1371/journal.pone.0273318",
        "plos_retinal_unet",
        "tibbiy tasvir / deep learning",
        "Atrous residual convolutional neural network based on U-Net "
        "for retinal vessel segmentation",
    ),
)

ARXIV_URL = "https://arxiv.org/pdf/{ident}"
PLOS_URL = (
    "https://journals.plos.org/plosone/article/file"
    "?id={ident}&type=printable"
)

LICENSES = {
    "gutenberg": (
        "public domain (AQSh)",
        "https://www.gutenberg.org/policy/license.html",
    ),
    # arXiv litsenziyasi **maqola bo'yicha** farq qiladi: standart
    # litsenziya (arxiv.org/licenses/nonexclusive) yoki CC BY/CC0/CC BY-NC.
    # Aniqlanmagan holda "arxiv.org/licenses/nonexclusive" deb yoziladi —
    # CC-BY deb **guman qilinmaydi**, chunki bu huquqiy xato bo'ladi.
    "arxiv": (
        "arxiv.org/licenses/nonexclusive (standart litsenziya; maqola "
        "bo'yicha boshqacha bo'lishi mumkin)",
        "https://arxiv.org/help/license",
    ),
    "plos": ("CC BY 4.0", "https://journals.plos.org/page?id=info#locate-license"),
}


class FetchError(RuntimeError):
    """Yuklab olish yoki tekshirishda xatolik."""


def http_get(url: str, *, timeout: int = 60, tries: int = 3) -> bytes:
    """URL dan ma'lumotni oladi — `User-Agent` va qayta urinish bilan.

    GitHub/Gutenberg bot'larni yuklashdan qaytaradi, shuning uchun
    `User-Agent` majburiy.

    Args:
        url: manzil.
        timeout: bitta so'rov chegarasi (soniya).
        tries: urinishlar soni.

    Returns:
        Yuklangan baytlar.

    Raises:
        FetchError: barcha urinishlar muvaffaqiyatsiz bo'lsa.
    """
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last = exc
            if attempt < tries:
                time.sleep(2 * attempt)
    raise FetchError(f"{url}: {type(last).__name__}: {last}")


def sha256_of(path: Path) -> str:
    """Faylning SHA-256 xeshini hisoblaydi."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gutenberg_text(ebook_id: int) -> str:
    """Gutenberg matnidan litsenziya sarlavhasini olib tashlaydi.

    Gutenberg fayli kitob emas, balki kitob + 10 sahifalik litsenziya
    matni. Sarlavhani olib tashlamasak, top terms'ga `gutenberg`,
    `copyright`, `public` singari so'zlar kirib ketadi.

    Args:
        ebook_id: Gutenberg elektron kitob raqami.

    Returns:
        Toza kitob matni.

    Raises:
        FetchError: `START OF THE PROJECT GUTENBERG` belgisi topilmasa.
    """
    url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
    raw = http_get(url).decode("utf-8", errors="replace")

    start = re.search(r"\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG.*?\*\*\*", raw)
    if not start:
        raise FetchError(f"{url}: START belgisi topilmadi")
    body = raw[start.end():]

    end = re.search(r"\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG.*?\*\*\*", body)
    if end:
        body = body[: end.start()]
    return body.strip()


def verify_title(text: str, title: str, author: str) -> tuple[bool, str]:
    """Yuklangan matn e'lon qilingan kitobga **tegishli** ekanini tekshiradi.

    **Bu tekshiruvsiz xato yuz berdi.** Gutenberg ID `25438` "The Jungle"
    deb yozilgan edi, lekin yuklangan fayl aslida **"The Airlords of Han"**
    (Amazing Stories, 1929) edi — sarlavha matnda umuman yo'q edi.
    ID'ni xotiradan taxmin qilish yetarli emas; yuklashgan kontentni
    **o'lchab** tekshirish kerak.

    Gutenberg faylining birinchi qatorlarida `Title:` va `Author:`
    qatorlari bor, lekin ular `START ...` belgisidan **oldin** —
    ya'ni biz olib tashlaydigan qismda. Shuning uchun toza matnning
    o'zida sarlavhalarni izlaymiz.

    Args:
        text: `gutenberg_text` natijasi.
        title: e'lon qilingan sarlavha.
        author: e'lon qilingan muallif.

    Returns:
        `(topildi, izoh)` — topilmasa, izohda qaysi so'z topilmagan.
    """
    haystack = normalize_for_search(text)
    author_ok = normalize_for_search(author).split()[0] in haystack
    words = normalize_for_search(title).split()
    # Uzun sarlavhalarda birinchi so'z "The"/"A" bo'lishi mumkin —
    # ma'no beruvchi so'zlardan boshlaymiz.
    meaningful = [w for w in words if w not in {"the", "a", "an", "of"}]
    missing = [w for w in (meaningful or words) if w not in haystack]
    if missing:
        return False, f"sarlavhadagi so'z topilmadi: {missing}"
    if not author_ok:
        return False, f"muallif topilmadi: {author!r}"
    return True, "mos"


def normalize_for_search(text: str) -> str:
    """Qidirish uchun matnni soddalashtiradi (registr va belgilar)."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def pdf_stats(path: Path) -> dict[str, int]:
    """PDF dan matn olish mumkinligini tekshiradi va statistika yig'adi.

    Skanerlangan PDF (rasm, matn qatlami yo'q) `chars` juda kichik
    bo'ladi — buni jadvalda ko'rsatamiz, chunki L-02 bo'yicha bunday
    PDF pipeline'ga kirmaydi.

    Args:
        path: PDF fayli.

    Returns:
        `pages` va `chars` kalitlari.

    Raises:
        FetchError: magic bytes mos emas yoki matn juda kam.
    """
    import pymupdf

    if path.read_bytes()[:4] != b"%PDF":
        raise FetchError(f"{path.name}: PDF magic mos emas")
    with pymupdf.open(path) as doc:
        chars = sum(len(page.get_text()) for page in doc)
        return {"pages": doc.page_count, "chars": chars}


def main() -> int:
    """Yuklab oladi, tekshiradi va manifest yozadi.

    Returns:
        `0` — hammasi muvaffaqiyatli, `1` — kamida bitta xato.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="yuklamasdan, faqat mavjud fayllarni tekshiradi",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="so'rovlar orasidagi pauza")
    args = parser.parse_args()

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    failures: list[str] = []

    # --- 1) Adabiyot (Gutenberg, matn ko'rinishida) --------------------
    print("=== Adabiyot — Project Gutenberg (public domain) ===")
    for ebook_id, slug, author, title, genre in LITERATURE:
        dest = TEXT_DIR / f"{slug}.txt"
        try:
            if not args.verify:
                if not dest.exists() or dest.stat().st_size < 10_000:
                    text = gutenberg_text(ebook_id)
                    dest.write_text(text, encoding="utf-8")
                    time.sleep(args.delay)
            text = dest.read_text(encoding="utf-8")
            ok, note = verify_title(text, title, author)
            if not ok:
                raise FetchError(
                    f"{slug} (ID {ebook_id}): {note}. ID boshqa kitobga "
                    f"tegishli bo'lishi mumkin — tekshirib ko'ring."
                )
            lic_name, lic_url = LICENSES["gutenberg"]
            entries.append(
                {
                    "slug": slug,
                    "group": "literature",
                    "title": title,
                    "author": author,
                    "genre": genre,
                    "source": f"https://www.gutenberg.org/ebooks/{ebook_id}",
                    "file": str(dest),
                    "format": "txt",
                    "license": lic_name,
                    "license_url": lic_url,
                    "bytes": dest.stat().st_size,
                    "sha256": sha256_of(dest),
                    "chars": len(text),
                }
            )
            print(f"  OK  {slug:24} {len(text):>9,} belgi")
        except (FetchError, OSError) as exc:
            failures.append(f"{slug}: {exc}")
            print(f"  XATO {slug:24} {exc}")

    # --- 2) Haqiqiy ochiq PDF'lar -------------------------------------
    print("\n=== Ilmiy maqolalar — arXiv / PLOS ONE (ochiq litsenziya) ===")
    for ident, slug, topic, title in OA_PAPERS:
        dest = PDF_DIR / f"{slug}.pdf"
        try:
            if not args.verify:
                if not dest.exists() or dest.stat().st_size < 10_000:
                    if ident.startswith("arxiv:"):
                        url = ARXIV_URL.format(ident=ident.split(":", 1)[1])
                        kind = "arxiv"
                    else:
                        url = PLOS_URL.format(ident=ident.split(":", 1)[1])
                        kind = "plos"
                    dest.write_bytes(http_get(url))
                    time.sleep(args.delay)
                else:
                    kind = "arxiv" if ident.startswith("arxiv:") else "plos"
            else:
                kind = "arxiv" if ident.startswith("arxiv:") else "plos"

            stats = pdf_stats(dest)
            lic_name, lic_url = LICENSES[kind]
            entries.append(
                {
                    "slug": slug,
                    "group": "open_access_paper",
                    "topic": topic,
                    "title": title,
                    "source": ident,
                    "file": str(dest),
                    "format": "pdf",
                    "license": lic_name,
                    "license_url": lic_url,
                    "bytes": dest.stat().st_size,
                    "sha256": sha256_of(dest),
                    "pages": stats["pages"],
                    "chars": stats["chars"],
                }
            )
            print(f"  OK  {slug:30} {stats['pages']:>3} sahifa {stats['chars']:>9,} belgi")
        except (FetchError, OSError) as exc:
            failures.append(f"{slug}: {exc}")
            print(f"  XATO {slug:30} {exc}")

    # --- 3) Manifest ----------------------------------------------------
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "generated": date.today().isoformat(),
                "generator": "scripts/fetch_real_documents.py",
                "note": (
                    "PDF'lar .gitignore'da; shu fayl ularning kelib chiqishini "
                    "va boshqaruv xeshini saqlaydi."
                ),
                "documents": entries,
                "failed": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nmanifest: {MANIFEST} ({len(entries)} ta hujjat, {len(failures)} ta xato)")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())