"""M3: matnni oldindan qayta ishlash testlari.

Test nomlari **xulqni** tushuntiradi (qa §63):
`fix_hyphenation_joins_word_split_across_lines`.

Muhim: testlar `data/raw/` ga **bog'liq emas** — u `.gitignore`da,
klonlangan repo'da bo'lmaydi. Barcha kirishlar test ichida yoziladi.

Uchta ustunli g'oya:
1. Har bir funksiya idempotent (`f(f(x)) == f(x)`).
2. Texnik atomalar (`AI`, `API`, `SQL`, `CNN`, `RAG`) saqlanadi.
3. Paragraf chegarasi `\\n\\n` saqlanadi (M4 chunking shuni kutadi).
"""

import pytest

from app.nlp import (
    PROTECTED_TERMS,
    STOP_WORDS,
    build_vocabulary,
    clean_pages,
    clean_text,
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
from app.schemas import PageText

#: Atomalar hech qanday tozalashdan keyin ham topilishi shart (README §M3).
TECH_TERMS = ["AI", "API", "SQL", "CNN", "RAG"]


# --- normalize_unicode -------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("eﬁle", "efile"),          # ligatura ﬁ -> fi
        ("ﬂow", "flow"),            # ligatura ﬂ -> fl
        ("№ 5", "No 5"),           # № -> No
        ("a\xa0b", "a b"),          # no-break space
        ("Café", "Café"),      # aksent saqlanadi
    ],
)
def test_normalize_unicode_expands_ligatures_and_normalizes(raw, expected):
    assert normalize_unicode(raw) == expected


def test_normalize_unicode_is_idempotent():
    raw = "eﬁle ﬂow № 5 a\xa0b"
    once = normalize_unicode(raw)
    assert normalize_unicode(once) == once


# --- fix_hyphenation ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("informa-\ntion", "information"),
        ("co-\noperate", "cooperate"),      # RFC: `co-operate` emas, `cooperate`
        ("state-\n  ful", "stateful"),     # keyingi qatorda bo'shliq bo'lishi mumkin
    ],
)
def test_fix_hyphenation_joins_word_split_across_lines(raw, expected):
    assert fix_hyphenation(raw) == expected


def test_fix_hyphenation_keeps_real_hyphenated_words():
    """`k-means` — defis bilan qator oxirida bo'lmasa, saqlanadi."""
    assert fix_hyphenation("k-means cluster") == "k-means cluster"


def test_fix_hyphenation_keeps_trailing_hyphen_when_next_line_starts_punct():
    """`-\\n` keyingi qator punktuatsiya bilan boshlanmasligi kerak,
    aks holda so'zni yopishtirish xato bo'ladi."""
    assert fix_hyphenation("word-\n- item") == "word-\n- item"


def test_fix_hyphenation_is_idempotent():
    once = fix_hyphenation("informa-\ntion")
    assert fix_hyphenation(once) == once


# --- remove_page_numbers -----------------------------------------------


@pytest.mark.parametrize(
    "page_num_line",
    ["12", "Page 12 of 40", "page 3 of 10", "12 / 40", "- 7 -", "  5  "],
)
def test_remove_page_numbers_deletes_standalone_number_line(page_num_line):
    text = f"Mavzu.\n\n{page_num_line}\n\nMatn davom etadi."
    result = remove_page_numbers(text)
    assert page_num_line.strip() not in result
    assert "Mavzu." in result
    assert "Matn davom etadi." in result


def test_remove_page_numbers_keeps_numbers_inside_sentences():
    """Faqat **butun qator** bo'lishi shart — gap ichidagi raqam matn."""
    text = "There are 3 layers and 2 filters."
    assert remove_page_numbers(text) == text


def test_remove_page_numbers_does_not_remove_line_starting_with_text():
    text = "12 million users signed up."
    assert remove_page_numbers(text) == text


def test_remove_page_numbers_is_idempotent():
    once = remove_page_numbers("Matn.\n\nPage 3 of 10")
    assert remove_page_numbers(once) == once


# --- remove_decorative_rules -------------------------------------------


@pytest.mark.parametrize("rule", ["=====", "-------", "*****", "~~~~~", "___"])
def test_remove_decorative_rules_deletes_rule_only_lines(rule):
    text = f"Sarlavha\n{rule}\nMatn."
    assert remove_decorative_rules(text) == "Sarlavha\nMatn."


def test_remove_decorative_rules_keeps_two_character_line():
    """`--` (markdown bullet) — dekorativ chiziq emas."""
    assert remove_decorative_rules("-- bullet") == "-- bullet"


def test_remove_decorative_rules_keeps_hyphenated_word_line():
    assert remove_decorative_rules("k-means is used") == "k-means is used"


def test_remove_decorative_rules_is_idempotent():
    once = remove_decorative_rules("Sarlavha\n=======\nMatn.")
    assert remove_decorative_rules(once) == once


# --- normalize_whitespace ----------------------------------------------


def test_normalize_whitespace_preserves_paragraph_break():
    """M4 chunking `\\n\\n` bo'yicha bo'linadi — chegarani yo'qotish mumkin emas."""
    result = normalize_whitespace("Paragraf bir.\n\n\n\nParagraf ikki.")
    assert result == "Paragraf bir.\n\nParagraf ikki."


def test_normalize_whitespace_collapses_horizontal_runs_and_nbsp():
    assert normalize_whitespace("a \xa0  b\tc") == "a b c"


def test_normalize_whitespace_strips_outer_whitespace():
    assert normalize_whitespace("\n\n  matn  \n\n") == "matn"


def test_normalize_whitespace_is_idempotent():
    once = normalize_whitespace("a \xa0 b\n\n\n\nc")
    assert normalize_whitespace(once) == once


# --- remove_repeated_headers_footers -----------------------------------


def _pages(*texts: str) -> list[PageText]:
    return [PageText(page=i + 1, text=t) for i, t in enumerate(texts)]


def test_remove_repeated_headers_footers_deletes_line_on_duplicate_pages():
    pages = _pages("Header\nMatn 1.", "Header\nMatn 2.", "Header\nMatn 3.")
    result = remove_repeated_headers_footers(pages)
    assert all("Header" not in p.text for p in result)


def test_remove_repeated_headers_footers_keeps_unique_content_line():
    """`Header` ketma-ket takrorlanadi, `Xulosa` esa — yo'q."""
    pages = _pages("Header\nXulosa", "Header\nMatn")
    result = remove_repeated_headers_footers(pages)
    assert "Xulosa" in result[0].text


def test_remove_repeated_headers_footers_keeps_long_line_above_max_chars():
    """Uzun takrorlanuvchi qator (masdal jadval sarlavhasi) — oddiy
    header emas, o'chirilmasligi kerak."""
    long_line = "Uzoq jadval sarlavhasi " * 4
    pages = _pages(f"{long_line}\nA", f"{long_line}\nB")
    result = remove_repeated_headers_footers(pages, max_chars=80)
    assert long_line in result[0].text


def test_remove_repeated_headers_footers_keeps_line_repeated_on_few_pages():
    pages = _pages("Header\nA", "Matn B", "Matn C", "Matn D")
    result = remove_repeated_headers_footers(pages, min_ratio=0.5)
    assert "Header" in result[0].text


def test_remove_repeated_headers_footers_never_empties_a_page():
    """Qisqa hujjatda chegara 1 ga tushsa, hujjat bo'shab ketmasin."""
    pages = _pages("A\nB", "A\nB")
    result = remove_repeated_headers_footers(pages)
    assert all(p.text.strip() for p in result)


def test_remove_repeated_headers_footers_counts_line_once_per_page():
    """Sahifa qisqa bo'lsa, `lines[:3]` va `lines[-3:]` ustma-ust tushadi —
    qortartib hisoblanmasligi kerak."""
    pages = _pages("A\nB", "A\nC")
    result = remove_repeated_headers_footers(pages, min_ratio=0.5)
    # "A" 2 marta takrorlanadi -> o'chadi, lekin sahifa bo'sh bo'lmaydi
    assert result[0].text.strip() == "B"
    assert result[1].text.strip() == "C"


def test_remove_repeated_headers_footers_preserves_page_numbers_and_order():
    pages = _pages("Header\nA", "Header\nB", "Header\nC")
    result = remove_repeated_headers_footers(pages)
    assert [p.page for p in result] == [1, 2, 3]


def test_remove_repeated_headers_footers_is_idempotent():
    pages = _pages("Header\nA", "Header\nB", "Header\nC")
    once = remove_repeated_headers_footers(pages)
    twice = remove_repeated_headers_footers(once)
    assert [p.text for p in once] == [p.text for p in twice]


def test_remove_repeated_headers_footers_handles_empty_list():
    assert remove_repeated_headers_footers([]) == []


# --- display_text / clean_text -----------------------------------------


def test_display_text_keeps_original_casing():
    """LLM va foydalanuvchiga ko'rsatiladi — registr saqlanadi."""
    assert display_text("The AI and the API.") == "The AI and the API."


def test_display_text_removes_all_artifacts_together():
    text = "eﬁle\ninforma-\ntion\n=====\n\nMatn.\xa0\n\nPage 4 of 9"
    result = display_text(text)
    assert "information" in result
    assert "efile" in result
    assert "=====" not in result
    assert "Page 4 of 9" not in result


def test_display_text_preserves_paragraphs_for_m4_chunking():
    assert display_text("Bir.\n\n\nIkki.") == "Bir.\n\nIkki."


def test_display_text_handles_empty_string():
    assert display_text("") == ""


def test_display_text_handles_whitespace_only_string():
    assert display_text("\n\n  \n") == ""


def test_display_text_is_idempotent():
    once = display_text("eﬁle\ninforma-\ntion\n=====\n\nMatn.\n\nPage 4 of 9")
    assert display_text(once) == once


def test_clean_text_is_alias_of_display_text():
    text = "Sarlavha\n=====\nMatn."
    assert clean_text(text) == display_text(text)


# --- ml_text ------------------------------------------------------------


def test_ml_text_lowercases_and_drops_stop_words():
    assert ml_text("The AI and the API are here.") == "ai api"


def test_ml_text_keeps_technical_terms():
    """README §M3 talabi: AI/API/SQL/CNN/RAG o'chmasligi shart."""
    text = "AI models call an API. SQL and CNN feed a RAG index."
    result = ml_text(text)
    for term in ["ai", "api", "sql", "cnn", "rag"]:
        assert term in result.split(), f"{term} yo'qoldi"


def test_ml_text_keeps_technical_terms_written_in_uppercase():
    result = ml_text("The AI, API, SQL, CNN and RAG systems.")
    for term in ["ai", "api", "sql", "cnn", "rag"]:
        assert term in result.split(), f"{term} yo'qoldi"


def test_ml_text_protects_system_word_that_sklearn_would_drop():
    """`sklearn` stop-list'i `system` ni o'chirdi (docs/limitations.md L-07)."""
    assert "system" in ml_text("The system has a cache.").split()


def test_ml_text_keeps_hyphenated_technical_token_intact():
    assert "k-means" in ml_text("We use K-Means clustering.").split()


def test_ml_text_keeps_token_plus_sign():
    assert "c++" in ml_text("Written in C++.").split()


def test_ml_text_does_not_emit_punctuation_only_tokens():
    """`Machine. Learning.` -> `['machine','learning']`; `.` so'z emas."""
    result = ml_text("Machine. Learning. Networks.").split()
    assert "." not in result
    assert result == ["machine", "learning", "networks"]


def test_ml_text_falls_back_to_lowercased_text_when_all_stop_words():
    """`one`, `two`, `three` — `sklearn` stop-list'ida. Foydali token
    qolmaganda bo'sh qaytarishdan ko'ra xom matnni qaytarish yaxshiroq.

    Fallback punktuatsiyani saqlaydi — bu ataylab qilingan: matnni
    butunlay tushirib qoldirishdan ko'ra foydaliroq.
    """
    assert ml_text("One. Two. Three.") == "one. two. three."
    assert ml_text("The and of") == "the and of"


def test_ml_text_handles_empty_string():
    assert ml_text("") == ""


def test_ml_text_is_idempotent():
    once = ml_text("The AI and the API are here. SQL and CNN feed a RAG index.")
    assert ml_text(once) == once


# --- himoya invariantlari ----------------------------------------------


def test_no_protected_term_leaks_into_stop_words():
    """Himoyalanadigan atoma `STOP_WORDS` ichida qolsa, `ml_text` uni
    o'chirib yuboradi — bu regressiya bo'lardi."""
    assert not (PROTECTED_TERMS & STOP_WORDS)


def test_stop_words_does_not_contain_common_english_function_words():
    for word in ["the", "and", "of", "is", "a", "in"]:
        assert word in STOP_WORDS


def test_protected_terms_covers_readme_required_terms():
    for term in ["ai", "api", "sql", "cnn", "rag"]:
        assert term in PROTECTED_TERMS


# --- clean_pages / ml_pages --------------------------------------------


def test_clean_pages_preserves_page_count_and_numbers():
    pages = _pages("Header\nMatn 1.", "Header\nMatn 2.", "Header\nMatn 3.")
    result = clean_pages(pages)
    assert [p.page for p in result] == [1, 2, 3]


def test_clean_pages_removes_headers_across_document():
    pages = _pages("Header\nMatn 1.", "Header\nMatn 2.", "Header\nMatn 3.")
    result = clean_pages(pages)
    assert all("Header" not in p.text for p in result)


def test_clean_pages_is_idempotent():
    """M4 `clean_pages` ni ikki marta ishlatishi mumkin — matn
    o'zgarmasligi kerak."""
    pages = _pages("Header\nMatn 1.", "Header\nMatn 2.", "Header\nMatn 3.")
    once = clean_pages(pages)
    twice = clean_pages(once)
    assert [p.text for p in once] == [p.text for p in twice]


def test_ml_pages_applies_ml_text_to_every_page():
    pages = _pages("Header\nThe AI is here.", "Header\nThe API is there.")
    result = ml_pages(pages)
    assert all("the" not in p.text.split() for p in result)
    assert "ai" in result[0].text.split()
    assert "api" in result[1].text.split()


def test_ml_pages_preserves_page_numbers():
    pages = _pages("Header\nAI rocks.", "Header\nAPI rules.", "Header\nSQL runs.")
    assert [p.page for p in ml_pages(pages)] == [1, 2, 3]


def test_pipeline_handles_empty_document():
    assert clean_pages([]) == []
    assert ml_pages([]) == []

# --- Real korpusdan olingan topilmalar (docs/limitations.md L-12) -------


class TestHyphenationVocabulary:
    """`fix_hyphenation` korpus lug'atiga qarab qaror qabul qiladi.

    O'lchov: 10 kitob + 5 ochiq maqola, 669 sahifa. Birlashtirish
    **470** nuqtada sinovdan o'tkazilganida **206** ta haqiqiy
    kompaniya (`book-shelves`, `twenty-four`, `cherry-tart`,
    `sequence-aligned`) buzilardi.
    """

    def test_broken_word_is_joined_when_corpus_confirms_it(self):
        assert fix_hyphenation("transduc-\ntion", frozenset({"transduction"})) == "transduction"

    def test_regex_captures_whole_fragments_not_one_char(self):
        """Regression: `(\\w)-` faqat oxirgi belgini ushlardi.

        `transduc-\\ntion` da `c-\\nt` match bo'lardi. Natija tasodufen
        to'g'ri chiqardi, shuning uchun bu xato faqat `vocabulary`
        bilan — ya'ni haqiqiy kompaniyani saqlash logikasi kiritilganda
        ko'rindi.
        """
        assert fix_hyphenation("transduc-\ntion") == "transduction"
        assert fix_hyphenation("Convolu-\ntional") == "Convolutional"
        # Oraliq belgi boshqa holatda buzilmasin.
        assert fix_hyphenation("self-suffi-\ncient") == "self-sufficient"

    def test_real_compound_is_kept_when_join_would_invent_a_word(self):
        """Asosiy regression test: `bookshelves` — bunday so'z yo'q."""
        vocab = build_vocabulary(["book-shelves on the shelf", "a cherry-tart"])
        assert fix_hyphenation("book-\nshelves", vocab) == "book-\nshelves"
        assert fix_hyphenation("cherry-\ntart", vocab) == "cherry-\ntart"

    @pytest.mark.parametrize(
        "text",
        ["twenty-\nfour", "star-\nfish", "rose-\ntree", "red-\nheaded", "flower-\npot"],
    )
    def test_dracula_compounds_survive(self, text):
        vocab = build_vocabulary([text.replace("-\n", "-"), "unrelated prose here"])
        assert fix_hyphenation(text, vocab) == text

    def test_scientific_compounds_survive(self):
        vocab = build_vocabulary(["sequence-aligned data", "position-wise attention"])
        assert fix_hyphenation("sequence-\naligned", vocab) == "sequence-\naligned"
        assert fix_hyphenation("position-\nwise", vocab) == "position-\nwise"

    def test_ligatures_are_fixed_before_hyphenation_check(self):
        """Regression: lug'at normalizatsiyasiz bo'lsa, ligaturali
        shakl (`suﬃcient`) solishtiriladigan shakl (`sufficient`)
        bilan hech qachon mos kelmaydi — va birlashtirish o'tkazib
        yuboriladi. `dracula.pdf` da 207 ta ligatura bor."""
        pages = _pages(
            "A suﬃcient amount of prose is required here.",
            "More prose mentioning suffi-\ncient in another line.",
            "Third page also says suffi-\ncient once more today.",
        )
        cleaned = "\n".join(p.text for p in clean_pages(pages))
        assert "suﬃcient" not in cleaned
        assert "suffi-\ncient" not in cleaned
        assert "sufficient" in cleaned

    def test_build_vocabulary_normalizes_ligatures(self):
        """Regression: lug'at `normalize_unicode` siz yig'ilmasin."""
        assert "office" in build_vocabulary(["oﬃce"])
        assert "ﬁsh" not in build_vocabulary(["oﬃce"])

    def test_fix_hyphenation_is_idempotent_with_vocabulary(self):
        vocab = build_vocabulary(["transduction", "book-shelves"])
        once = fix_hyphenation("transduc-\ntion and book-\nshelves", vocab)
        assert fix_hyphenation(once, vocab) == once

    def test_match_is_case_insensitive_but_case_is_preserved(self):
        """`Convolu-` + `tional` -> `Convolutional` (katta harf saqlanadi)."""
        vocab = frozenset({"convolutional"})
        assert fix_hyphenation("Convolu-\ntional", vocab) == "Convolutional"

    def test_without_vocabulary_everything_is_joined(self):
        """Bitta matnda ishlanganda eski xatti-harakat saqlanadi."""
        assert fix_hyphenation("book-\nshelves") == "bookshelves"
        assert fix_hyphenation("book-\nshelves", frozenset()) == "book-\nshelves"

    def test_build_vocabulary_is_lowercased(self):
        assert build_vocabulary(["Hello World"]) == frozenset({"hello", "world"})

    def test_build_vocabulary_accepts_a_generator(self):
        assert build_vocabulary(p for p in ["alpha beta"]) == frozenset({"alpha", "beta"})

    def test_build_vocabulary_on_empty_input(self):
        assert build_vocabulary([]) == frozenset()

    def test_clean_pages_uses_corpus_vocabulary(self):
        """`clean_pages` kompaniyani saqlashi kerak — sahifa alohida
        tozalanganda buzilardi."""
        pages = _pages(
            "Dracula keeps a book-\nshelves and twenty-\nfour windows.",
            "This prose mentions book-shelves and twenty-four again.",
            "More prose here about cherry-tart and star-fish pies.",
        )
        joined = "\n".join(p.text for p in clean_pages(pages))
        assert "bookshelves" not in joined
        assert "twentyfour" not in joined
        assert "cherrytart" not in joined

    def test_clean_pages_still_joins_confirmed_words(self):
        """So'z korpusda biror joyda **butun** ko'rinishi shart."""
        pages = _pages(
            "The transduc-\ntion step runs here.",
            "Later the full transduction appears in running prose.",
            "And transduc-\ntion shows up again in this text.",
        )
        joined = "\n".join(p.text for p in clean_pages(pages))
        assert "transduc-\ntion" not in joined
        assert "transduction" in joined

    def test_always_hyphenated_word_is_not_joined(self):
        """Korpusda hech qayerda butun ko'rinmasa — birlashtirilmaydi.

        Bu ataylab qoldirilgan cheklov: lug'at tekshiruvi ishonchli
        manba, lekin u faqat **korpusda tasdiqlangan** shakllarni
        birlashtiradi. Xavfsiz yo'l — birlashtirmasdan qolish
        (`twenty-four` ham to'g'ri token).
        """
        pages = _pages(*["The transduc-\ntion step runs here."] * 3)
        assert "transduc-\ntion" in "\n".join(p.text for p in clean_pages(pages))


class TestEdgeLinesNone:
    """`edge_lines=None` o'lchov bilan aniqlangan **xavfli** rejim.

    `arxiv_bradley_terry.pdf` da 509 qator, jumladan formula jadvali
    belgilari (`X` 67 marta, `i` 56, `1` 43) yo'q qilinadi. Shuning
    uchun default `3` qoladi va `clean_pages` undan foydalanmaydi.
    """

    def test_default_stays_at_three(self):
        """Faqat `Header` dan iborat sahifalar bo'shab ketmasin —
        xavfsizlik himoyasi ishlashi kerak."""
        pages = _pages(*["Header\nMatn"] * 10)
        result = remove_repeated_headers_footers(pages)
        assert all(p.text for p in result)
        assert all("Matn" in p.text for p in result)

    def test_none_finds_header_in_the_middle_of_a_page(self):
        """PLOS holati: header matn tartibida sahifa o'rtasida.

        Sahifalar **turli** bo'lishi shart — aks holda `edge_lines=None`
        da har bir qator takrorlanuvchi bo'lib, sahifa bo'shab qoladi
        va xavfsizlik himoyasi asl matnni qaytaradi.
        """
        pages = [
            PageText(page=n, text=f"Sahifa {n}\n" + "\n".join(
                [f"{n}-qator {i}" for i in range(20)] + ["Header"] + [f"{n}-qator {i}" for i in range(20, 40)]
            ))
            for n in range(1, 4)
        ]
        assert all("Header" in p.text for p in pages)
        # Chekka qarab izlash topmaydi — header 21-qatorda.
        assert all("Header" in p.text for p in remove_repeated_headers_footers(pages, edge_lines=3))
        # `None` topadi.
        cleaned = remove_repeated_headers_footers(pages, edge_lines=None)
        assert all("Header" not in p.text for p in cleaned)
        assert all(len(p.text) > 0 for p in cleaned)

    def test_none_is_not_the_default_used_by_clean_pages(self):
        """`clean_pages` o'rta qatordagi header ni o'chirmasligi kerak —
        bu PLOS kabi ilmiy PDF larda formula belgilarini o'chirish
        xavfi bilan birga keladi."""
        body = [f"Qator {i}" for i in range(40)]
        pages = [
            PageText(page=i + 1, text="Bosh\n" + "\n".join(body[:20] + ["Header"] + body[20:]))
            for i in range(4)
        ]
        assert all("Header" in p.text for p in clean_pages(pages))

    def test_none_keeps_short_tokens_that_are_real_content(self):
        """Xavfning o'zi: `X`, `i`, `1` — formula jadvali belgilari."""
        body = "\n".join(f"X i 1 {n}" for n in range(30))
        pages = [PageText(page=n, text=body) for n in range(1, 11)]
        assert all("X" in p.text for p in remove_repeated_headers_footers(pages, edge_lines=None))
