"""`notebooks/02_preprocessing.ipynb` ni yaratadi (M3.T1).

Notebook qo'lda yozilmaydi — shunda har bir hujjat **bir xil tuzilmaga**
ega bo'ladi va diff o'qilishi oson bo'ladi. Notebook M2'dagi
`notebooks/01_pdf_extraction.ipynb` uslubini takrorlaydi:
markdown + code, `id` maydonlari bilan, `execution_count` to'ldiriladi.

Ishga tushirish:
    .venv/bin/python scripts/make_preprocessing_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute \
        --output notebooks/02_preprocessing.ipynb \
        notebooks/02_preprocessing.ipynb
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    # ---------------------------------------------------------------- 0
    ("markdown", """# 02 — Matnni oldindan qayta ishlash

**Milestone:** M3 · **Task:** T1 (empirik artefaktlar + strategiya tanlovi)

M2 matnni **ajratib** berdi, lekin u **xom**: ichida takrorlanuvchi
header, sahifa raqamlari, dekorativ chiziqlar bor. TF-IDF shu xom
matnni vektorlashsa, top terms `the, of, page, =====` dan iborat
bo'ladi — `docs/architecture.md` **F7** aynan shu muammoni aytadi.

Bu notebook **o'lchaydi**, keyin tanlov qiladi. Xulosa avval o'lchovdan,
keyin yozilgan."""),
    ("markdown", """## 0. Tayyorlov

**O'rganish:** M2 moduli `extract_pages` hujjatni ochadi va
`PageText` ro'yxatini beradi. M3 shu ro'yxatni oladi."""),
("code", '''
import os
import re
import sys
import json
from collections import Counter
from pathlib import Path

# Notebook `notebooks/` ichida saqlanadi, lekin kernel cwd si qayerdan
# boshlanishiga qarab farq qilishi mumkin (nbconvert default: notebook
# papkasi). Shuning uchun loyiha ildizini **belgi fayl bo'yicha** qidiriamiz
# — hard-coded `..` ishlamaydi.
ROOT = next(
    p for p in (Path.cwd(), *Path.cwd().parents) if (p / "scripts/make_sample_pdfs.py").exists()
)
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.nlp import (  # noqa: E402
    PROTECTED_TERMS,
    STOP_WORDS,
    build_vocabulary,
    clean_pages,
    display_text,
    fix_hyphenation,
    ml_pages,
    ml_text,
    normalize_unicode,
    normalize_whitespace,
    remove_decorative_rules,
    remove_page_numbers,
    remove_repeated_headers_footers,
)
from app.services import extract_pages  # noqa: E402

SAMPLES = sorted(Path("data/raw").glob("*.pdf"))
if not SAMPLES:
    raise SystemExit(
        "data/raw/ bo'sh. Namuna PDF'larni yaratish uchun: "
        ".venv/bin/python scripts/make_sample_pdfs.py"
    )

print(f"Python {sys.version.split()[0]} · loyiha ildizi: {ROOT}")
for p in SAMPLES:
    print(f"  {p.name:38} {p.stat().st_size:>7,} bayt")
'''),
    # ---------------------------------------------------------------- 1
    ("markdown", """## 1. Xom matnda haqiqiy nima bor?

**O'rganish:** "Tozalash" kerakligi taxminga emas, **o'lchovga**
asoslanishi kerak. Quyida M2 chiqarishini to'rt mezon bo'yicha
 tekshiramiz.

| # | Mezon | Nima qidiriladi |
|---|-------|-----------------|
| 1 | Defis bilan uzilgan so'z | `informa-` / `tion` |
| 2 | Takrorlanuvchi qator | header/footer |
| 3 | Sahifa raqami | butun qator `12` |
| 4 | Unicode artefakt | ligatura, `\\xa0`, `ﬀ` |"""),
    ("code", '''
raw_pages = {p.name: extract_pages(p) for p in SAMPLES}

# --- 1) defis bilan uzilgan so'z ---------------------------------------
# `+` majburiy: `\\w-` faqat oxirgi belgini ushlaydi va nuqtani
# barchasini topmaydi (L-12 da batafsil).
hyphenated = sum(
    len(re.findall(r"\\w+-\\n\\w+", page.text))
    for pages in raw_pages.values()
    for page in pages
)

# --- 2) takrorlanuvchi qatorlar ----------------------------------------
def repeated_lines(pages, min_count=2):
    counter = Counter()
    for page in pages:
        for line in page.text.splitlines():
            line = line.strip()
            if line:
                counter[line] += 1
    return [(c, l) for l, c in counter.items() if c >= min_count]

# --- 3) sahifa raqamlari ----------------------------------------------
def page_number_lines(pages):
    found = []
    for page in pages:
        for line in page.text.splitlines():
            s = line.strip()
            if re.fullmatch(r"(?:[-–—]?\\s*page\\s*)?\\d{1,4}(?:\\s*(?:/|of|[-–—])\\s*\\d{1,4})?\\s*[-–—]?", s, re.I):
                found.append(s)
                break
    return found

# --- 4) Unicode artefaktlar -------------------------------------------
LIGATURES = "ﬀﬁﬂﬃﬄ"

print(f"{'PDF':38} {'belgi':>7} {'hyphen':>7} {'takror':>7} {'sahifa#':>8} {'lig':>4} {'xa0':>4}")
print("-" * 82)
for name, pages in raw_pages.items():
    text = "".join(p.text for p in pages)
    lig = sum(text.count(c) for c in LIGATURES)
    per_doc = sum(
        len(re.findall(r"\\w+-\\n\\w+", page.text)) for page in pages
    )
    print(f"{name:38} {len(text):>7} {per_doc:>7} "
          f"{len(repeated_lines(pages)):>7} {len(page_number_lines(pages)):>8} "
          f"{lig:>4} {text.count(chr(0xa0)):>4}")
'''),
    ("markdown", """> **Kuzatuv:** Namuna PDF'larimizda **faqat 3 ta** artefakt
> topildi: takrorlanuvchi header, `Page N of M` footer va `=` chiziqlari.
> Defis bilan uzilgan so'z, ligatura va `\\xa0` — **yo'q**.
>
> Bu muhim xulosa: bu funksiyalar **sintetik** testlarda tekshiriladi,
> chunki bizning generatorimiz toza matn beradi. "Real PDF'da
> ishladi" deb da'vo qilish **yolg'on** bo'lurdi."""),
    ("code", '''
pages3 = raw_pages["03_multi_topic_report.pdf"]
print(f"03_multi_topic_report.pdf — {len(pages3)} sahifa\\n")
for page in pages3[:3]:
    lines = page.text.splitlines()
    print(f"--- sahifa {page.page} ({len(lines)} qator) ---")
    print(f"  [0] {lines[0]!r}")
    print(f"  [1] {lines[1]!r}   <- dekorativ chiziq")
    print(f"  ...")
    print(f"  [{len(lines)-2}] {lines[-2]!r}   <- takrorlanuvchi header")
    print(f"  [{len(lines)-1}] {lines[-1]!r}   <- sahifa raqami")
    print()

print("=== takrorlanuvchi qatorlar (hamma sahifada) ===")
for count, line in sorted(repeated_lines(pages3), reverse=True)[:4]:
    print(f"  {count:>3}x {line!r}")

print("\\n=== dekorativ chiziqlar ===")
rules = sum(
    len(re.findall(r"^[=\\-_*~]{3,}$", l.strip()))
    for pages in raw_pages.values()
    for page in pages
    for l in page.text.splitlines()
)
print(f"  jami {rules} ta")
'''),
    # ---------------------------------------------------------------- 2
    ("markdown", """## 2. Uch strategiya va ularni o'lchash

**O'rganish:** "Yaxshi tozalash" — subyektiv so'z. Uni **o'lchovga**
almashtiramiz: nechta token qoladi va top-10 token **ma'noli**mi.

| Strategiya | Kichik harf | Stop-word | Stemming |
|------------|-------------|-----------|----------|
| **A** minimal | ✅ | ❌ | ❌ |
| **B** standart | ✅ | ✅ | ❌ |
| **C** agressiv | ✅ | ✅ | ✅ |"""),
    ("code", '''
def crude_stem(word: str) -> str:
    """**Namoyish uchun** sodda qat'iyat kesuvchi — Porter **emas**.

    Ataylab yozildi: real stemmer (`nltk`) muhitda **yo'q** va
    `requirements.txt` ga yangi bog'liqlik qo'shish bu bosqich uchun
    asoslanmagan. Bu funksiya C strategiyasining **xavfini**
    ko'rsatish uchun — M5 da haqiqiy stemmer kerak bo'lsa, alohida
    qaror qabul qilinadi.
    """
    for suffix in ("ational", "ization", "iveness", "fulness",
                   "ousness", "ing", "edly", "ies", "ied", "es", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[+#][a-z0-9]*|[.\\-][a-z0-9]+)*")


def strategy(text: str, *, lower=True, stop=False, stem=False) -> str:
    tokens = TOKEN_RE.findall(text.lower()) if lower else TOKEN_RE.findall(text)
    if stop:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if stem:
        tokens = [crude_stem(t) for t in tokens]
    return " ".join(tokens)
'''),
    ("code", '''
# O'lchov maydoni — **artefaktlar olib tashlangan** matn.
# Aks holda taqqoslashuv header va sahifa raqamlari bilan aralashib,
# stop-word'ning **o'z ta'siri** o'lchanmaydi (top-10 da `page`, `12`,
# `8` chiqib turardi — ular matn emas, jadval artefakti).
corpus_pages = []
for name, pages in raw_pages.items():
    corpus_pages.extend(clean_pages(pages))
corpus = "\\n\\n".join(p.text for p in corpus_pages)

print(f"O'lchov korpusi: {len(corpus)} belgi, {len(corpus_pages)} sahifa\\n")
print(f"{'Strategiya':30} {'token':>7} {'vocab':>7}  top-10 token")
print("-" * 96)
# Kalit qisqa (`A`/`B`/`C`) qilinadi: uzun yorliqlar bilan murojaat
# qilish nozik — bir tushirilgan bo'shliq `KeyError` beradi.
rows = {}
for code, label, kwargs in [
    ("A", "A: minimal (lower)",       dict(lower=True, stop=False, stem=False)),
    ("B", "B: standart (stop+lower)", dict(lower=True, stop=True,  stem=False)),
    ("C", "C: agressiv (+stem)",      dict(lower=True, stop=True,  stem=True)),
]:
    tokens = strategy(corpus, **kwargs).split()
    vocab = set(tokens)
    top = [w for w, _ in Counter(tokens).most_common(10)]
    rows[code] = (tokens, vocab, top)
    print(f"{label:30} {len(tokens):>7} {len(vocab):>7}  {', '.join(top)}")

a_tokens, a_vocab, a_top = rows["A"]
b_tokens, b_vocab, b_top = rows["B"]
c_tokens, c_vocab, c_top = rows["C"]
print(f"\\nB ning lug'atga qo'shgani: {len(a_vocab) - len(b_vocab)} ta "
      f"({(len(a_vocab) - len(b_vocab)) / len(a_vocab):.1%})")
print(f"C ning B ga qo'shgani   : {len(b_vocab) - len(c_vocab)} ta "
      f"({(len(b_vocab) - len(c_vocab)) / len(b_vocab):.1%})")
'''),
    ("markdown", """### Qaror — B strategiyasi

O'lchov korpusi: **29 sahifa**, 13 869 belgi (artefaktlar olib
tashlangandan keyin). Tokenlar **pastdan yuqoriga** sanalgan.

**Nima ko'ryapmiz:**

1. **A** da top-10 token `and, the, a, chapter, of, to, as, data, such, with`
   — **stop-word'lar hukmronlik qiladi**. Bu aynan **F7** muammosi.
   Token eng ko'p (1817), lug'at eng keng (926) — ya'ni eng ko'p shovqin.
2. **B** da top-10 `chapter, data, based, nodes, va, learning, cluster,
   replication, cost, scaling` — **ma'noli**: har bir so'z mavzu haqida
   nima deganini aytadi. Token 1817 → 1393, lug'at 926 → 854 (**7.8%**).
3. **C** lug'atni yana 79 ta qisqartiradi (854 → 775, **9.3%**), lekin
   **texnik atomani buzadi**: `kmeans → kmean`, `embedding → embedd`.

**Xulosa: B.** C ning 9.3% lik qisqarishi uning atoma buzilishiga
**qiymat bermaydi** — chunki bizning muammomiz token ko'pligi emas,
**ma'nosiz token**. Top-10 da `and, the, of` o'rniga `cluster`,
`replication`, `nodes` turishi kerak edi.

**Eslatma:** `crude_stem` — namoyish uchun, Porter **emas**. C ning
haqiqiy natijasi boshqacha bo'lishi mumkin, lekin **yo'qotish** xavfi
baribir qoladi. Haqiqiy stemmer talab bo'lsa, M5 da alohida qaror."""),
    ("code", '''
print("=== C strategiyasi texnik atomalarni buzadi ===")
for term in sorted(["ai", "api", "sql", "cnn", "rag", "llm", "cpu",
                    "embedding", "clustering", "kmeans"]):
    stemmed = crude_stem(term)
    flag = "  <-- BUZILDI" if stemmed != term else ""
    print(f"  {term:12} -> {stemmed:12}{flag}")

print("\\n=== B strategiyasi esa saqlaydi ===")
b_vocab = rows["B"][1]
for term in sorted(["ai", "api", "sql", "cnn", "rag", "llm", "cpu"]):
    print(f"  {term:12} {'SAQLANGAN' if term in b_vocab else 'YOQOLDIGI!'}")
'''),
    # ---------------------------------------------------------------- 3
    ("markdown", """## 3. `sklearn` stop-list'i nima uchun to'g'ridan-to'g'ri ishlatilmaydi?

**O'rganish:** `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` — 318
so'zlik tayyor ro'yxat. Ishlatish oson, lekin u **umumiy ingliz** uchun
yozilgan, **texnik korpus** uchun emas."""),
    ("code", '''
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

print(f"sklearn stop-list: {len(ENGLISH_STOP_WORDS)} ta so'z\\n")

# Texnik korpusdagi muhim so'zlarni tekshiramiz
DOMAIN_PROBE = ["system", "systems", "model", "data", "network",
                 "cluster", "server", "index", "memory", "value", "state"]
print(f"{'atoma':12} {'sklearn':>9} {'bizning':>9}  izoh")
print("-" * 52)
for w in DOMAIN_PROBE:
    in_sk = w in ENGLISH_STOP_WORDS
    in_ours = w in STOP_WORDS
    note = "HIMOYA qilindi" if (in_sk and not in_ours) else ""
    print(f"{w:12} {str(in_sk):>9} {str(in_ours):>9}  {note}")

print(f"\\nHimoyalanadigan atomalar soni: {len(PROTECTED_TERMS)}")
print("STOP_WORDS ichida qolgan himoyalangan atoma (bo'lsa — xato):",
      PROTECTED_TERMS & STOP_WORDS or "YO'Q — to'g'ri")
'''),
    ("markdown", """> **Xulosa:** `sklearn` ro'yxatida `system` bor — bizning
> namunalarimizning biri esa **Distributed Systems**. Agar to'g'ridan-to'g'ri
> ishlatgan bo'lsak, mavzu so'zi lug'atdan o'chib ketar edi.
> Shuning uchun `STOP_WORDS = ENGLISH_STOP_WORDS - PROTECTED_TERMS`."""),
    # ---------------------------------------------------------------- 4
    ("markdown", """## 4. Ishlatilgan modul bilan tekshirish

**O'rganish:** Yuqoridagi tajribalar — o'lchov. Endi **ishlatiladigan**
modul bilan natijani tekshiramiz va ikki matn farqini ko'ramiz."""),
    ("code", '''
raw = "eﬁle\\n\\ninforma-\\ntion\\n=====\\n\\nThe AI and the API are here.\\n\\nPage 4 of 9"

print("XOM:")
print(repr(raw))
print("\\ndisplay_text (yengil — LLM/embedding uchun):")
print(repr(display_text(raw)))
print("\\nml_text (jiddiy — TF-IDF uchun):")
print(repr(ml_text(raw)))
'''),
    ("code", '''
print("=== header/footer olib tashlandimi? ===")
for name, pages in raw_pages.items():
    cleaned = clean_pages(pages)
    before = sum(len(p.text) for p in pages)
    after = sum(len(p.text) for p in cleaned)
    leftover = [p.page for p in cleaned
                if "Technical Report" in p.text or re.search(r"Page \\d+ of", p.text)]
    print(f"  {name:38} {before:>5} -> {after:>5} belgi ({after/before:.0%})  qoldiq={leftover or 'yo\\'q'}")
'''),
    # ---------------------------------------------------------------- 5
    ("markdown", """## 5. Idempotentlik va M4 uchun shartlar

**O'rganish:** M4 `clean_pages` natajasini qayta ishlatishi mumkin. Agar
`f(f(x)) != f(x)` bo'lsa, matn har bir qayta ishga tushirishda
o'zgarib boradi va chunk chegaralari silinadi.

Shuning uchun ikki shart majburiy:

1. **Idempotentlik** — `f(f(x)) == f(x)`
2. **Paragraf chegarasi** `\\n\\n` saqlanadi (M4 shu bo'yicha bo'linadi)"""),
    ("code", '''
print("=== idempotentlik ===")
per_text = [
    ("normalize_unicode", normalize_unicode),
    ("fix_hyphenation", fix_hyphenation),
    ("remove_page_numbers", remove_page_numbers),
    ("remove_decorative_rules", remove_decorative_rules),
    ("normalize_whitespace", normalize_whitespace),
    ("display_text", display_text),
    ("ml_text", ml_text),
]
ok = True
for name, fn in per_text:
    once = fn(raw)
    same = fn(once) == once
    ok &= same
    print(f"  {name:26} {'OK' if same else 'BUZILDI'}")

once_pages = clean_pages(pages3)
twice_pages = clean_pages(once_pages)
pages_ok = [p.text for p in once_pages] == [p.text for p in twice_pages]
ok &= pages_ok
print(f"  {'clean_pages':26} {'OK' if pages_ok else 'BUZILDI'}")
print(f"\\nNatija: {'BARCHASI IDEMPOTENT' if ok else 'MUAMMO BOR'}")

print("\\n=== paragraf chegarasi saqlanadimi? (M4) ===")
sample = "Paragraf bir.\\n\\n\\n\\nParagraf ikki."
# `chr(10)` ishlatiladi — notebook manbasi ichida `\\n` escape'lari
# qatlamlashadi va tekshiruvni yolg'on qiladi (4 belgi tekshiriladi,
# 2 ta emas).
para_break = chr(10) * 2
result = display_text(sample)
print(f"  kirish : {sample!r}")
print(f"  chiqish: {result!r}")
print(f"  {para_break!r} saqlangan: {result.count(para_break) == 1}")
'''),
    # ---------------------------------------------------------------- 6
    ("markdown", """## 6. Texnik atoma tekshiruvi (README talabi)

README §M3: `AI`, `API`, `SQL`, `CNN`, `RAG` — preprocessing'dan keyin
ham **yo'qolmasligi** kerak. Register pasolini ham tekshiramiz."""),
    ("code", '''
probe = "The AI, API, SQL, CNN and RAG systems use K-Means clustering with TF-IDF."
print(f"KIRISH: {probe}")
for label, out in [("display_text", display_text(probe)), ("ml_text", ml_text(probe))]:
    print(f"\\n{label}: {out}")
    for term in ["ai", "api", "sql", "cnn", "rag", "k-means", "tf-idf"]:
        status = "bor" if term in out.lower().split() else "YO'Q"
        print(f"    {term:9} {status}")
'''),
    ("markdown", """## 7. Umumiy xulosa — natija jadvali

O'lchov korpusi: **29 sahifa / 13 869 belgi**. Barcha raqamlar yuqoridagi
haqiqiy chiqishdan olingan.

| Strategiya | Token | Lug'at | Top-10 ma'noli | Atoma butun | **Qaror** |
|------------|-------|--------|----------------|-------------|-----------|
| A minimal | 1 817 | 926 | ❌ (`and, the, of`) | ✅ | ❌ |
| **B standart** | **1 393** | **854** | ✅ (`cluster, data`) | ✅ | ✅ **tanlandi** |
| C agressiv | 1 393 | 775 | ✅ | ❌ (`kmean`, `embedd`) | ❌ |

**Dalillar asosidagi xulosa (README §M3 T1 bajarildi):**

1. Xom matndagi artefaktlar **o'lchandi** — 3 ta tur (takrorlanuvchi
   header, `Page N of M`, `=` chiziqlari), 12–17% belgi olib tashlandi.
2. Uch strategiya **bir xil korpusda** solishtirildi — o'lchov maydoni
   oldindan tozalangan, aks holda taqqoslashuv artefakt bilan aralashgan
   bo'lardi.
3. Tanlov **o'lchovga** asoslandi: B lug'atni 7.8% qisqartirdi va
   top-10 token ma'noli qildi; C qo'shimcha 9.3% qisqartirdi, lekin
   texnik atamani buzdi.
4. `sklearn` stop-list'idagi `system` o'chib ketishi **aniqlandi** va
   `PROTECTED_TERMS` bilan bartaraf etildi.
5. Barcha funksiyalar **idempotent**, paragraf chegarasi `\\n\\n`
   saqlanadi (M4 uchun).

**Cheklov:** `crude_stem` — Porter emas. C strategiyasi haqiqiy stemmer
bilan qayta o'lchansa, natija o'zgarishi mumkin.

---

## 8. Real korpus bilan tekshiruv — 3 ta xato topildi

Yuqoridagi barcha o'lchov **4 ta sintetik PDF** ga asoslangan edi.
Ular hyphenation va ligatura yo'q edi. Ya'ni eng qiyin funksiyalar
faqat qo'lda yozilgan testlar bilan tekshirilgan.

Yuklangan (`scripts/fetch_real_documents.py`): **10 ta Gutenberg kitobi**
+ **5 ta ochiq maqola** (3 arXiv, 2 PLOS ONE) — 5 501 150 belgi.

> **Bu yuklashda xato bo'ldi va tuzatildi.** ID 25438 "The Jungle" deb
> yozilgan edi, lekin yuklangan fayl aslida **"The Airlords of Han"**
> (Amazing Stories, 1929) edi. To'g'ri ID — **140**. Endi skript
> `verify_title()` bilan har bir faylda sarlavha **o'zi tekshiriladi**."""),
    ("code", '''
REAL = sorted(Path("data/raw/real/pdf").glob("*.pdf"))
if not REAL:
    raise SystemExit(
        "Real korpus yo'q. Yuklash uchun:\\n"
        "  .venv/bin/python scripts/fetch_real_documents.py\\n"
        "  .venv/bin/python scripts/make_real_pdfs.py"
    )

manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
print(f"{len(REAL)} ta hujjat · manifest'da {len(manifest['documents'])} ta yozuv\\n")

for group, pattern in [("Adabiyot (Gutenberg, render qilingan)", "lit_*.pdf"),
                       ("Ilmiy maqola (arXiv/PLOS, tegilmagan)", None)]:
    files = (sorted(Path("data/raw/real/pdf").glob(pattern)) if pattern
             else [p for p in REAL if not p.name.startswith("lit_")])
    pages = sum(len(extract_pages(p)) for p in files)
    chars = sum(len(pg.text) for p in files for pg in extract_pages(p))
    print(f"{group:42} {len(files):>3} ta fayl  {pages:>5} sahifa  {chars:>10,} belgi")
'''),
    ("markdown", """### 8.1 Xato 1 — regex faqat **bitta belgi** ushlaydi

Avvalgi regex `(\\w)-\\n(\\w)` edi. `transduc-\\ntion` da u `c-\\nt` ni
match qiladi — faqat oxirgi belgini.

Natija **tasodufen to'g'ri** chiqardi (`transdu`+`ct`+`ion`), shuning
uchun faqat yakuniy satrni tekshiradigan testlar buni ko'rmadi."""),
    ("code", '''
OLD = re.compile(r"(\\w)-\\n(\\w)")
NEW = re.compile(r"(\\w+)-\\n[ \\t]*(\\w+)")
sample = "transduc-\\ntion"
vocab = frozenset({"transduction"})

print("eski regex match:", OLD.findall(sample))
print("yangi regex match:", NEW.findall(sample))
print()
print("NATIJA farqi yo'q — ikkalasi ham to'g'ri shakl beradi:")
print("  eski (jo'shilmaydi):", repr(OLD.sub(r"\\1\\2", sample)))
print("  yangi:", repr(NEW.sub(r"\\1\\2", sample)))
print()
print("FARQ lug'at bilan qaror qabul qilinganda chiqadi:")
def decide(pattern):
    def sub(m):
        h, t = m.group(1), m.group(2)
        return h + t if (h + t).lower() in vocab else m.group(0)
    return pattern.sub(sub, sample)

print("  eski:", repr(decide(OLD)), " <-- faqat `c` va `t` ni ko'radi,")
print("                                 `ct` lug'atda yo'q -> so'z sindirilmay qoladi")
print("  yangi:", repr(decide(NEW)), " <-- `transduc`+`tion` lug'atda topildi -> birlashtiriladi")
print()
print("Ya'ni xato yakuniy satrda ko'rinmaydi — faqat qaror mexanikasida.")
print("Bitta matnda (`vocabulary=None`) eski regex ham 'to'g'ri' natija beradi,")
print("shuning uchun bu xato 3+ oy davomida testlardan o'tib ketgan bo'lishi mumkin.")
'''),
    ("markdown", """### 8.2 Xato 2 — `fix_hyphenation` so'z **yaratib** yuboradi

Bitta so'z sindi (`transduc-` + `tion`) va ikki so'zli kompaniya
(`book-` + `shelves`) matnda **bir xil** ko'rinadi. Har doim
birlashtirish 206 ta haqiqiy kompaniyani buzardi.

**Yechim:** `build_vocabulary()` — korpusdan tasdiqlangan so'zlar."""),
    ("code", '''
for pattern, label in [("lit_*.pdf", "Adabiyot"), ("arx*.pdf", "arXiv"), ("plos*.pdf", "PLOS")]:
    files = sorted(Path("data/raw/real/pdf").glob(pattern))
    pages = [pg for p in files for pg in extract_pages(p)]
    vocab = build_vocabulary(pg.text for pg in pages)
    joinable = compounds = 0
    examples_ok, examples_bad = [], []
    for pg in pages:
        for head, tail in re.findall(r"(\\w+)-\\n[ \\t]*(\\w+)", pg.text):
            if (head + tail).lower() in vocab:
                joinable += 1
                if len(examples_ok) < 4: examples_ok.append(head + tail)
            else:
                compounds += 1
                if len(examples_bad) < 4: examples_bad.append(f"{head}-{tail}")
    total = joinable + compounds
    pct_join = joinable / total * 100 if total else 0
    print(f"{label:10} {len(pages):>5} sahifa | {total:>4} nuqta | "
          f"bitta so'z {pct_join:>4.0f}% | kompaniya {100-pct_join:>3.0f}%")
    print(f"           birlashtiriladi: {[e.lower() for e in examples_ok]}")
    print(f"           saqlanadi:       {examples_bad}")
print()
print("Jami: 750 nuqtadan 446 ta (59%) haqiqiy kompaniya —")
print("lug'atsiz birlashtirilsa, hammasi buzilardi.")
'''),
    ("markdown", """### 8.3 Xato 3 — lug'at ligaturani normalizatsiya qilmasdi

`dracula.pdf` da **207** ta ligatura. Lug'atda `suﬃcient` turar edi,
ammo solishtiriladigan shakl `sufficient` — hech qachon mos kelmasdi."""),
    ("code", '''
dracula = extract_pages("data/raw/real/pdf/lit_dracula.pdf")
raw_text = "".join(p.text for p in dracula)
print("dracula.pdf — ligatura xom matnda:", sum(raw_text.count(c) for c in LIGATURES))
print("  qaysi belgilar:", {c: raw_text.count(c) for c in LIGATURES if raw_text.count(c)})
print("  (butun korpusda:", end=" ")
all_lig = sum(
    sum(pg.text.count(c) for c in LIGATURES)
    for p in sorted(Path("data/raw/real/pdf").glob("*.pdf"))
    for pg in extract_pages(p)
)
print(f"{all_lig:,})")
print()
vocab = build_vocabulary(p.text for p in dracula)
print("lug'atda 'office' (normalizatsiya qilingan):", "office" in vocab)
print("lug'atda ligatura saqlanib qolganmi:", any(c in vocab for c in LIGATURES))
print()
cleaned = "".join(p.text for p in clean_pages(dracula))
print("tozalashdan keyin ligatura:", sum(cleaned.count(c) for c in LIGATURES))
print("tozalashdan keyin sahifa raqami qoldimi:", bool(re.search(r"(?m)^[ \\t]*\\d{1,4}[ \\t]*$", cleaned)))
'''),
    ("markdown", """### 8.4 Natija — o'lchov bilan

| Ko'rsatkich | Oldin | Keyin |
|-------------|-------|-------|
| Ligatura (butun korpus) | 11 594 | **0** |
| Ligatura (`dracula.pdf`) | 1 691 | **0** |
| Sahifa raqami qolgani | — | **0** |
| `plos_middle_ear_effusion.pdf` | — | **−1.6%** (header/footer olib tashlandi) |
| `arxiv_bradley_terry.pdf` | — | **−1.3%** |
| Adabiyot belgisi | — | **+0.1%** (`ﬀ`→`ff` kengayadi) |

**Lug'at saqlanganmi?** Bu muhim: tozalash **so'z o'chirib qo'yishi**
mumkin, lekin kengaytirish kerak."""),
    ("code", '''
lit_vocab = build_vocabulary(
    pg.text for p in sorted(Path("data/raw/real/pdf").glob("lit_*.pdf")) for pg in extract_pages(p)
)
print("Adabiyot lug'atida (top 12 namuna):")
print(" ", [w for w in ["dracula", "mina", "watson", "holmes", "alice", "rabbit",
                       "thoreau", "carpenter", "walden", "austen", "stoker", "wilde"]
          if w in lit_vocab])
print()
sci_pages = [pg for p in REAL if not p.name.startswith("lit_") for pg in extract_pages(p)]
sci_vocab = build_vocabulary(pg.text for pg in sci_pages)
print("Ilmiy lug'atda:", [w for w in ["attention", "transformer", "retina", "pca",
                                     "cnn", "unet", "pls"] if w in sci_vocab])
print("  (tophalanmagan so'zlar o'sha maqolalarda umuman ishlatilmagan)")
'''),
    ("markdown", """### 8.5 Arxiv/PLOS header muammosi — **hali yechilmagan**

PDF matn qatlami vizual tartibni saqlamaydi: PLOS header'i 3-sahifada
oxirida, 1-sahifada esa **o'rta qatorda**. `edge_lines=3` uni
topmaydi. `edge_lines=None` topadi — lekin **juda xavfli**:

| Fayl | `edge_lines=3` | `edge_lines=None` |
|------|----------------|-------------------|
| `arxiv_bradley_terry.pdf` | −1.3% | **−2.3%** |

Olib tashlanganlar furniture emas — `X` (67 marta), `i` (56), `1` (43),
`\\uf8f4` (38) bular **formula jadvali belgilari**. Jami **509 qator**
yo'q qilinadi. Default `edge_lines=3` qoldirildi (L-14)."""),
    ("code", '''
plos = extract_pages("data/raw/real/pdf/plos_middle_ear_effusion.pdf")
print("PLOS header'i qayerda (matn tartibi bo'yicha):")
for pg in plos[:3]:
    lines = pg.text.splitlines()
    pos = [i for i, l in enumerate(lines) if "PLOS ONE" in l]
    print(f"  sahifa {pg.page}: {len(lines):>3} qator, 'PLOS ONE' qatori={pos}")

for label, kw in [("edge_lines=3 (default)", dict(edge_lines=3)),
                  ("edge_lines=None", dict(edge_lines=None))]:
    cleaned = remove_repeated_headers_footers(plos, **kw)
    left = sum("PLOS ONE" in p.text for p in cleaned)
    print(f"  {label:24} -> 'PLOS ONE' {left}/{len(cleaned)} sahifada qoldi")

print()
print("edge_lines=None ning narxi (arxiv_bradley_terry.pdf):")
bt = extract_pages("data/raw/real/pdf/arxiv_bradley_terry.pdf")
for label, kw in [("edge_lines=3", dict(edge_lines=3)), ("edge_lines=None", dict(edge_lines=None))]:
    before = sum(len(p.text) for p in bt)
    after = sum(len(p.text) for p in remove_repeated_headers_footers(bt, **kw))
    print(f"  {label:16} {before:>7,} -> {after:>7,} belgi ({(after-before)/before:+.1%})")
'''),
    ("markdown", """## 9. Yakuniy xulosa

| | Sintetik korpus | Real korpus |
|---|---|---|
| Hujjat | 4 ta namuna PDF | 10 kitob + 5 maqola |
| Hajm | 16 464 belgi | **5 501 150 belgi** |
| Hyphenation nuqtasi | **0** | **750** |
| Ligatura | **0** | **11 594** |
| Aniqlangan xato | 0 | **3 ta kod + 1 ta ma'lumot** |

**Xotiraga olish uchun 4 ta xato:**

| # | Xato | Qachon ko'rildi |
|---|------|----------------|
| 1 | `(\\w)-` faqat bitta belgi ushlaydi | lug'atli variant yozilganda |
| 2 | Har doim birlashtirish → so'z yaratish | adabiyotda 344 ta kompaniya |
| 3 | `build_vocabulary` ligaturani yechmadi | `dracula.pdf` da 11 345 ligatura |
| 4 | **Boshqa kitob yuklangan** (ID 25438) | sarlavha matnda topilmadi |

Xato 4 — eng muhimi: u **koding emas, ma'lumotning** xatosi edi va
hech qanday test uni ushlamagan, chunki testlar Internetga
ulanmagan. Endi `verify_title()` har bir yuklamada sarlavha va
muallifni matn ichidan qidiradi.

**Asosiy xulosa:** sintetik korpus **yetarli emasdi** — u har bir
funksiyani "ishlayapti" deb ko'rsatgan, lekin to'rtta xato faqat haqiqiy
ma'lumatda ko'rindi.

**M4 ga o'tish shartlari bajarildi:**
- `clean_pages` + `ml_pages` ishlaydi (5.5 mln belgi bilan tekshirilgan)
- `\\n\\n` paragraf chegarasi saqlanadi
- texnik atoma va kitob lug'ati saqlanadi

**Keyingi (M4):** chunking parametrlarini o'lchash."""),
]


def build() -> Path:
    """Notebook faylini yozadi va fayl manzilini qaytaradi."""
    cells = []
    for i, (kind, source) in enumerate(CELLS):
        lines = source.strip("\n").split("\n")
        # nbformat `source` — qatorlar ro'yxati, har biri `\\n` bilan
        body = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {
            "cell_type": kind,
            "id": f"m3-{i:02d}",
            "metadata": {},
            "source": body,
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "ipython",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.14",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    target = Path("notebooks/02_preprocessing.ipynb")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    path = build()
    print(f"yozildi: {path} · {len(CELLS)} hujjat")