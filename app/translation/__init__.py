"""Tarjima moduli — pluggable, default ochiq (ADR-003).

Ishlatish:

```python
from app.core.config import get_settings
from app.translation import get_translator

translator = get_translator(get_settings())
translated = translator.translate_pages(page_texts, target="en")
```

Chiqish:

```python
from app.translation import (
    GoogleTranslateTranslator,
    NoopTranslator,
    TranslationError,
    TranslationUnavailableError,
    get_translator,
    validate_batch_mode,
)
```

Qoidalar:
- Faqat `document` yoki `page` darajasi. **Chunk level taqiqlangan.**
- Default `NoopTranslator` — tashqi so'rov yuborilmaydi.
"""

from app.translation.base import (
    VALID_BATCH_MODES,
    BatchMode,
    Translator,
    validate_batch_mode,
)
from app.translation.exceptions import (
    TranslationConfigError,
    TranslationError,
    TranslationResponseError,
    TranslationUnavailableError,
)
from app.translation.factory import PROVIDERS, get_translator
from app.translation.google_translate import GoogleTranslateTranslator
from app.translation.noop import NoopTranslator

__all__ = [
    "BatchMode",
    "GoogleTranslateTranslator",
    "NoopTranslator",
    "PROVIDERS",
    "TranslationConfigError",
    "TranslationError",
    "TranslationResponseError",
    "TranslationUnavailableError",
    "Translator",
    "VALID_BATCH_MODES",
    "get_translator",
    "validate_batch_mode",
]