"""Matnni oldindan qayta ishlash — `docs/architecture.md` FR3, FR4.

M2 sahifa matnini beradi, lekin u **xom**: defis bilan uzilgan so'zlar,
takrorlanuvchi header/footer, sahifa raqamlari, dekorativ chiziqlar.
Ushbu modul ularni olib tashlaydi va ikki xil matn beradi:

| Chiqish | Kim ishlatadi | Tozalash |
|---------|---------------|----------|
| `display_text` | Embedding (M9), LLM (M11), foydalanuvchi | yengil — o'qishga tayyor |
| `ml_text` | TF-IDF (M5) | jiddiy — vektorlashga tayyor |

**Nega ikkalasi kerak:** bitta matnni ikkala maqsadga ishlatish mumkin
emas. LLM'ga `lowercase` + stop-word'siz matn bersang, tabiiy tilni
yo'qotadi; TF-IDF'ga to'zalashlanmagan matn bersang, `the, of, page` dan
iborat top terms chiqadi (`docs/architecture.md` F7).

**Texnik atomalar himoyasi.** `AI`, `API`, `SQL`, `CNN`, `RAG`, `KMeans`,
`TF-IDF`, `LLM` — bular ma'lumot tashiydi va **hech qachon** olib
tashlanmasligi shart. Shuning uchun `sklearn` ning stop-word ro'yxati
to'g'ridan-to'g'ri ishlatilmaydi: unda `system` kabi so'zlar bor
(o'lchangan — `docs/limitations.md` L-07). `STOP_WORDS` shu ro'yxatdan
`PROTECTED_TERMS` chiqarib olingan.

**Idempotentlik:** har bir funksiya `f(f(x)) == f(x)` ga qarab
loyihalangan. Aks holda `clean_pages(clean_pages(pages))` matnni
yana o'zgartiradi va M4 chunking manbalarini buzadi.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.schemas.document import PageText

__all__ = [
    "PROTECTED_TERMS",
    "STOP_WORDS",
    "build_vocabulary",
    "clean_pages",
    "clean_text",
    "display_text",
    "fix_hyphenation",
    "ml_pages",
    "ml_text",
    "normalize_unicode",
    "normalize_whitespace",
    "remove_decorative_rules",
    "remove_page_numbers",
    "remove_repeated_headers_footers",
]

#: Texnik atomalar — stop-word filtridan **har doim** himoyalangan.
#: Kengaytirish kerak bo'lsa shu yerga qo'shiladi (test ham tekshiradi).
PROTECTED_TERMS: frozenset[str] = frozenset({
    # umumiy
    "ai", "ml", "api", "sql", "cnn", "rnn", "llm", "rag", "gpu", "cpu",
    "nlp", "ocr", "etl", "sdk", "ide", "url", "uri", "http", "json", "xml",
    "yaml", "csv", "nosql",
    # usullar / modellar
    "kmeans", "kmeans++", "bert", "gpt", "lstm", "svm", "tfidf",
    "tf", "idf", "bm25", "mds", "pca", "umap", "tsne", "lda", "naive",
    "bayes", "xgboost", "lightgbm", "random", "forest",
    # domen so'zlari
    "system", "systems", "model", "models", "data", "database", "databases",
    "server", "servers", "cluster", "clusters", "clustering", "network",
    "networks", "node", "nodes", "cache", "buffer", "queue", "index",
    "schema", "middleware", "kernel", "runtime", "daemon", "protocol",
    "throughput", "latency", "replication", "sharding", "partition",
})

#: Stop-word ro'yxati — `sklearn` dan olingan, texnik atomalar chiqarilgan.
STOP_WORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS) - PROTECTED_TERMS

#: Token regex. `C++`, `k-means`, `tf-idf`, `3.14` — bitta token; `words.` — `words`.
#: Birinchi belgi har doim harf/raqam bo'lishi shart, aks holda punctuation
#: (`a`, `.`, `...`) so'z sifatida o'tib ketadi.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[+#][a-z0-9]*|[.\-][a-z0-9]+)*")

#: Satir oxiridagi so'z uzilishi: `transduc-` + `tion`.
#:
#: **`+` majburiy.** Birinchi versiya `(\w)-` edi — `transduc-\ntion` da
#: `c-\nt` ni match qilib, faqat oxirgi belgini olardi. Natija
#: tasodufen to'g'ri chiqardi (`transdu`+`ct`+`ion`), shuning uchun
#: faqat yakuniy satrni tekshiradigan testlar bu xatoni ko'rmadi —
#: xato `build_vocabulary` bilan ishlatilganda paydo bo'ldi.
_HYPHEN_BREAK_RE = re.compile(r"(\w+)-\n[ \t]*(\w+)")

#: Butun qatorga qo'llaniladi: `12`, `Page 12 of 40`, `12 / 40`, `- 7 -`.
#: Kirish/chiqish defisi ixtiyoriy — aks holda `- 7 -` qolib ketadi.
_PAGE_NUM_LINE_RE = re.compile(
    r"^[ \t]*(?:[-–—][ \t]*)?(?:page[ \t]*)?\d{1,4}"
    r"(?:[ \t]*(?:/|of|[-–—])[ \t]*\d{1,4})?"
    r"(?:[ \t]*[-–—])?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


# --- Bir qatorli normalizatsiyalar -------------------------------------


def normalize_unicode(text: str) -> str:
    """Unicode'ni `NFKC` shakliga keltiradi.

    Ligatura (`ﬁ`, `ﬂ`) oddiy ASCII'ga yoyiladi, `№` -> `No`, turli
    bo'shliqlar (`\\xa0`, `\\u202f`) bir xil ko'rinadi.

    Args:
        text: xom matn.

    Returns:
        `NFKC` normalizatsiyasidan o'tgan matn.

    Example:
        >>> normalize_unicode("e\\uFB01le")
        'efile'
    """
    return unicodedata.normalize("NFKC", text)


def build_vocabulary(pages: Iterable[str]) -> frozenset[str]:
    """Matndan kichik harfli tokenlar to'plamini yig'adi.

    `fix_hyphenation` uchun kerak: birlashtirilgan shakl korpusda
    **boshqa yerda** uchrayotganmi tekshirish uchun.

    **Nega `normalize_unicode` majburiy.** Lug'at birlashtirishdan
    **keyin** solishtiriladi, `display_text` esa avval normalizatsiya
    qiladi. Normalizatsiyasiz lug'atda `suﬃcient` (ligatura bilan)
    turadi, ammo solishtiriladigan shakl `sufficient` — hech qachon
    topilmaydi. `dracula.pdf` da 207 ta ligatura bor, ya'ni bu
    real miqdarda.

    Args:
        pages: matnlar (sahifa matnlari ro'yxati ham bo'ladi).

    Returns:
        Tokenlar to'plami.

    Example:
        >>> build_vocabulary(["Oﬃce WORK"]) == frozenset({"office", "work"})
        True
        >>> "ﬁsh" in build_vocabulary(["oﬃce"])
        False
    """
    vocab: set[str] = set()
    for text in pages:
        vocab.update(_TOKEN_RE.findall(normalize_unicode(text).lower()))
    return frozenset(vocab)


def fix_hyphenation(text: str, vocabulary: frozenset[str] | None = None) -> str:
    """Satir oxiridagi defis bilan uzilgan so'zni birlashtiradi.

    PDF matn qatlami so'zlarni satr oxirida `informa-` / `tion` ko'rinishida
    saqlaydi. Join qilinmasdan, bu **bitta so'z ikki bo'lak** bo'lib
    ko'rinadi va lug'atga ikki xil shakl kiradi.

    **Nega `vocabulary` kerak.** Haqiqiy kitoblarda (`docs/limitations.md`
    L-12) defis bilan uzilgan so'zning ikki xil turi bor:

    | Tur | Namuna | Join natijasi |
    |-----|--------|---------------|
    | Bitta so'z sindi | `transduc-` + `tion` | `transduction` ✅ |
    | Ikki so'z, defisli | `book-` + `shelves` | `bookshelves` ❌ |

    O'lchov (10 ta kitob + 5 ta ochiq maqola, 669 sahifa): birlashtirish
    **470** nuqtasida sinovdan o'tkazilganida **206** ta haqiqiy kompaniya
    buzilardi (`bookshelves`, `cherrytart`, `twentyfour`,
    `sequencealigned`). Shuning uchun `vocabulary` berilgan bo'lsa,
    faqat **ro'yxatda bor** shakllar birlashtiriladi.

    **Nega bu xavfsiz yo'l.** Xatolar **nomutanosib**:
    keraksiz birlashtirish -> so'z **yo'q** paydo bo'ladi (lug'atga
    iflos token kiradi); birlashtirmaslik -> defisli token saqlanadi
    (bu ham to'g'ri token). Yomonroq holatni yo'q qilish kerak.

    Args:
        text: xom matn.
        vocabulary: `build_vocabulary` chiqarishi mumkin bo'lgan
            tokenlar to'plami. `None` bo'lsa — hamma joyda
            birlashtiriladi (faqat bitta matnda ishlash uchun).

    Returns:
        `so'z-\\nso'z` -> `so'zso'z` qilingan matn.

    Example:
        >>> fix_hyphenation("informa-\\ntion")
        'information'
        >>> fix_hyphenation("book-\\nshelves", frozenset({"bookshelves"}))
        'bookshelves'
        >>> fix_hyphenation("book-\\nshelves", frozenset({"book"}))
        'book-\\nshelves'
    """
    if vocabulary is None:
        return re.sub(_HYPHEN_BREAK_RE, r"\1\2", text)

    def _join(match: re.Match[str]) -> str:
        head, tail = match.group(1), match.group(2)
        if (head + tail).lower() in vocabulary:
            return head + tail
        return match.group(0)

    return re.sub(_HYPHEN_BREAK_RE, _join, text)


def remove_page_numbers(text: str) -> str:
    """Yolg'iz qator turib qolgan sahifa raqamlarini olib tashlaydi.

    `12`, `- 12 -`, `Page 12 of 40`, `12 / 40` kabi qatorlar matn emas —
    bu jadval yoki sarlavha **qismi** bo'lishi mumkin, shuning uchun
    faqat butun qatorga qat'iy qaror qabul qilinadi.

    Args:
        text: xom matn.

    Returns:
        Sahifa raqami qatorlari olib tashlangan matn.

    Example:
        >>> remove_page_numbers("Matn.\\n\\nPage 3 of 10\\n")
        'Matn.'
    """
    text = _PAGE_NUM_LINE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def remove_decorative_rules(text: str) -> str:
    """Dekorativ chiziq qatorlarini olib tashlaydi.

    Sarlavha ostidagi `=====` yoki `-----` chiziqlari matn **yo'q** —
    ular lug'atga `=====` kabi shaklda kirib, top terms'ni ifloslaydi.
    Namuna PDF'larimizda bular 3-12 marta sahifa boshida uchraydi
    (`docs/limitations.md` L-08).

    Args:
        text: xom matn.

    Returns:
        Chiziq qatorlari olib tashlangan matn.

    Example:
        >>> remove_decorative_rules("Sarlavha\\n=======\\nMatn.")
        'Sarlavha\\nMatn.'
    """
    lines = text.split("\n")
    kept = [
        line
        for line in lines
        if not re.fullmatch(r"[ \t]*[=\-_*~]{3,}[ \t]*", line)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def normalize_whitespace(text: str) -> str:
    """Bo'shliqlarni normallashtiradi, **paragraf chegarasini saqlaydi**.

    M4 chunking `\\n\\n` bo'yicha bo'linadi, shuning uchun qator
    chegaralarini oddiy bo'shliqqa almashtirish **chunk chegarasini
    buzadi**.

    Args:
        text: xom matn.

    Returns:
        Bo'shliqlari tekislangan matn.

    Example:
        >>> normalize_whitespace("a \\xa0 b\\n\\n\\n\\nc")
        'a b\\n\\nc'
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- Sahifalararo tozalash ---------------------------------------------


def remove_repeated_headers_footers(
    pages: list[PageText],
    *,
    min_ratio: float = 0.5,
    max_chars: int = 80,
    edge_lines: int | None = 3,
) -> list[PageText]:
    """Takrorlanuvchi header/footer qatorlarini olib tashlaydi.

    **Nega sahifalar ro'yxati kerak:** bitta sahifani ko'rib, qator
    takrorlanuvchimi yoki yo'qmi aniqlab bo'lmaydi — buni faqat hujjat
    darajasida ko'rish mumkin.

    Uchta shart **ham** bajarilishi kerak:
    1. qator sahifa **cheggasida** (`edge_lines` ichida)
    2. qator `min_ratio` dan ko'p sahifada uchraydi
    3. qator `max_chars` dan qisqa

    Chekga sharti muhim: aks holda haqiqiy takrorlanuvchi *mavzu*
    (masalan har sahifada "Methodology") matndan butunlay o'chib ketadi.

    **Kamida 2 marta** talab qo'yiladi. Aks holda qisqa hujjatda
    (masalan 2 sahifa, `0.5 * 2 = 1`) chegarasi 1 ga tushib, har bir
    qator "takrorlanuvchi" bo'lib qoladi va hujjat **bo'shab ketadi**.

    Args:
        pages: sahifalar (kirish tartibida saqlanadi).
        min_ratio: takrorlanish chegarasi (0..1).
        max_chars: qatorning maksimal uzunligi.
        edge_lines: sahifa boshida va oxirida nechta qator tekshiriladi.
            `None` berilsa — qator **hammasi** tekshiriladi.

            **Ogohlantirish: `None` xavfli, o'lchov bilan aniqlangan.**
            PDF matn qatlami vizual tartibni saqlamaydi, shuning uchun
            PLOS header'i 3-sahifada oxirida, 1-sahifada esa **o'rta
            qatorda** chiqadi va chekka qarab izlash uni topmaydi
            (`docs/limitations.md` L-12). `None` topadi — lekin

            | Fayl | `edge_lines=3` | `edge_lines=None` |
            |------|----------------|-------------------|
            | `arxiv_bradley_terry.pdf` | −1.3% | **−2.3%** |
            | `plos_middle_ear_effusion.pdf` | −1.7% | −4.3% |

            va olib tashlanganlar **furniture emas**: `X` (67 marta),
            `i` (56), `1` (43), `\uf8f4` (38) — bular ilmiy maqoladagi
            formula jadvali belgilari. Jami **509 qator** yo'q qilindi.

            Xulosa: `None` **yoqilmaydi**, `clean_pages` undan foydalanmaydi.
            To'g'ri tuzatish — M2 dan blok koordinatalarini (`get_text("blocks")`)
            olish va `y` joylashuviga qarab filtrlash; bu M3 doirasida emas.

    Returns:
        Tozalangan yangi sahifalar ro'yxati.

    Example:
        >>> from app.schemas.document import PageText
        >>> pages = [
        ...     PageText(page=1, text="Sarlavha\\nHeader\\nMatn 1."),
        ...     PageText(page=2, text="Sarlavha\\nHeader\\nMatn 2."),
        ... ]
        >>> [p.text for p in remove_repeated_headers_footers(pages)]
        ['Matn 1.', 'Matn 2.']
    """
    if not pages:
        return []

    def edge_indexes(lines: list[str]) -> list[int]:
        n = len(lines)
        if edge_lines is None:
            return list(range(n))
        return sorted(set(range(min(edge_lines, n))) | set(range(max(0, n - edge_lines), n)))

    def is_edge(index: int, total: int) -> bool:
        if edge_lines is None:
            return True
        return index < edge_lines or index >= total - edge_lines

    counts: Counter[str] = Counter()
    for page in pages:
        lines = page.text.splitlines()
        # Indekslar to'plami — qisqa sahifada `lines[:3]` va `lines[-3:]`
        # ustma-ust tushib, qatorni juftlab hisoblardi.
        for i in edge_indexes(lines):
            stripped = lines[i].strip()
            if stripped and len(stripped) <= max_chars:
                counts[stripped] += 1

    threshold = max(2, min_ratio * len(pages))
    repeated = {line for line, count in counts.items() if count >= threshold}
    if not repeated:
        return list(pages)

    cleaned: list[PageText] = []
    for page in pages:
        lines = page.text.splitlines()
        n = len(lines)
        kept = [
            line
            for i, line in enumerate(lines)
            if not (line.strip() in repeated and is_edge(i, n))
        ]
        # Xavfsizlik: agar tozalash sahifani butunlay bo'shatib yuborsa,
        # o'zgartirilmagan shaklni qaytaramiz.
        if not any(line.strip() for line in kept):
            cleaned.append(page)
            continue
        cleaned.append(
            PageText(page=page.page, text=re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip())
        )
    return cleaned


# --- Yuqori darajali funksiyalar ---------------------------------------


def display_text(text: str, vocabulary: frozenset[str] | None = None) -> str:
    """Foydalanuvchi, LLM va embedding uchun matn (yengil tozalash).

    Registr, punktatsiya va atomalar **saqlanadi** — bu matn
    foydalanuvchiga ko'rsatiladi va LLM'ga yuboriladi.

    Args:
        text: xom sahifa matni.
        vocabulary: `build_vocabulary` natijasi. Berilsa, hyphenation
            faqat korpusda tasdiqlangan shakllarda birlashtiriladi
            (`fix_hyphenation` ga qarang).

    Returns:
        O'qishga tayyor matn.

    Example:
        >>> display_text("Sarlavha\\n=====\\nAI va API. ")
        'Sarlavha\\nAI va API.'
    """
    text = normalize_unicode(text)
    text = fix_hyphenation(text, vocabulary)
    text = remove_page_numbers(text)
    text = remove_decorative_rules(text)
    return normalize_whitespace(text)


def ml_text(text: str, vocabulary: frozenset[str] | None = None) -> str:
    """TF-IDF va clustering uchun matn (jiddiy tozalash).

    `display_text` ga qo'shimcha: kichik harfga o'tkaziladi va
    **stop-word** lar olib tashlanadi. Texnik atomalar
    (`PROTECTED_TERMS`) saqlanadi.

    Strategiya tanlovi o'lchov bilan asoslangan
    (`notebooks/02_preprocessing.ipynb`): A strategiyasida top-10 so'z
    `and, the, a, of, to` dan iborat edi; C (stemming) esa lug'atni
    7.5% ga qisqartirgan holda `kmeans`->`kmean`, `embedding`->`embedd`
    kabi atamalarni buzdi.

    Args:
        text: xom sahifa matni yoki `display_text` natijasi.
        vocabulary: `build_vocabulary` natijasi (korpus darajasida).

    Returns:
        Vektorlashga tayyor matn.

    Example:
        >>> ml_text("The AI and the API are here.")
        'ai api'
    """
    cleaned = display_text(text, vocabulary)
    tokens = _TOKEN_RE.findall(cleaned.lower())
    kept = [t for t in tokens if t not in STOP_WORDS]
    if not kept:
        return cleaned.lower()
    return " ".join(kept)


def clean_text(text: str) -> str:
    """`display_text` ning izchil nomi (README §M3 shabloni).

    Args:
        text: xom matn.

    Returns:
        Tozalangan matn.

    See Also:
        [`display_text`] — bir xil natija, semantik nom bilan.
    """
    return display_text(text)


def clean_pages(pages: list[PageText]) -> list[PageText]:
    """Barcha sahifalarni tozalaydi, jumladan header/footer ni.

    Korpus darajasindagi ikki qaror **bu yerda** qabul qilinadi:

    1. `build_vocabulary` — barcha sahifalardan lug'at yig'iladi va
       `fix_hyphenation` ga beriladi. Bittasiz (har sahifa alohida)
       ishlasak, haqiqiy kompaniyalar (`book-shelves`, `twenty-four`)
       buziladi — o'lchov: 470 nuqtadan 206 tasini.
    2. Header/footer barcha sahifalar ko'rilgandan keyin topiladi.

    Args:
        pages: M2 dan olingan sahifalar.

    Returns:
        Tozalangan sahifalar.
    """
    vocabulary = build_vocabulary(p.text for p in pages)
    stripped = [
        PageText(page=p.page, text=display_text(p.text, vocabulary))
        for p in pages
    ]
    return remove_repeated_headers_footers(stripped)


def ml_pages(pages: list[PageText]) -> list[PageText]:
    """Barcha sahifalarni `ml_text` ga o'tkazadi.

    Args:
        pages: M2 dan olingan sahifalar.

    Returns:
        Vektorlashga tayyor sahifalar.
    """
    vocabulary = build_vocabulary(p.text for p in pages)
    cleaned = clean_pages(pages)
    return [PageText(page=p.page, text=ml_text(p.text, vocabulary)) for p in cleaned]