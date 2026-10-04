"""NLP moduli — matnni oldindan qayta ishlash (`docs/architecture.md` FR3/FR4).

Ishlatish:

```python
from app.nlp import clean_pages, display_text, ml_text

# Foydalanuvchi / LLM / embedding uchun
matn = display_text("Sarlavha\\n=====\\nAI va API.")

# TF-IDF va clustering uchun
matn = ml_text("The AI and the API are here.")   # -> 'ai api here.'

# Butun hujjat (M2 dan kelgan sahifalar)
toza = clean_pages(pages)
```

Qoidalar:
- Har bir funksiya **idempotent**: `f(f(x)) == f(x)`.
- Texnik atomalar (`AI`, `API`, `SQL`, `CNN`, `RAG`) **hech qachon**
  olib tashlanmaydi — `PROTECTED_TERMS`.
- Paragraf chegarasi `\\n\\n` saqlanadi — M4 chunking shu bo'yicha
  bo'linadi.
- Default strategiya **B** (lowercase + stop-word, stemming yo'q).
  Tanlov o'lchov bilan asoslangan: `notebooks/02_preprocessing.ipynb`.
"""

from app.nlp.preprocessing import (
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