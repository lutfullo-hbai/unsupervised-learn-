"""Tarjima moduli xatolari.

Barcha tarjima xatolari `TranslationError` dan meros oladi, shuning uchun
`app/` ning boshqa qatlamlari bitta `except TranslationError` bilan
tarjimani ushlaydi.

Ishlatiladigan mapping (docs/io_contract.md §3):
- `TranslationUnavailableError` -> HTTP 503 (ML natijasi saqlanadi)
- `TranslationConfigError`      -> HTTP 500 (noto'g'ri konfiguratsiya)
- `TranslationResponseError`    -> HTTP 503 (provider javobi buzilgan)
"""


class TranslationError(Exception):
    """Tarjima moduli boshlang'ich xatosi."""


class TranslationUnavailableError(TranslationError):
    """Provider ishlamadi: timeout, tarmoq xatosi yoki 5xx.

    Bu xato vaqtinchalik deb hisoblanadi, shuning uchun
    `GoogleTranslateTranslator` uni cheklangan miqdorda qayta urinadi.
    """


class TranslationConfigError(TranslationError):
    """Konfiguratsiya noto'g'ri: noma'lum provider yoki taqiqlangan batch rejimi.

    Bu xato ishlash vaqti emas, sozlash vaqti xatosi — qayta urinish
    ma-nosiz, shuning uchun retry qilinmaydi.
    """


class TranslationResponseError(TranslationError):
    """Provider javobi kutilgan formatda emas.

    Masalan: JSON parse xatosi, `data[0]` segmentlar yo'q yoki
    segment ichidagi tarjima matni string emas.
    """