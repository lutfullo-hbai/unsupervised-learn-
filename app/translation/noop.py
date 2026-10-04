"""Passthrough tarjimachi — moduli default provayderi.

Tashqi so'rov **_absolute yuborilmaydi**. Bu local-first tamoyilining
ifodasi: `TRANSLATION_ENABLED=false` (default) bo'lganda pipeline butunlay
offline ishlaydi.

Natija kirish bilan **bir xil** bo'lib chiqadi, shuning uchun M12 uni
pipeline'ga ulashda alohida branch kerak emas.
"""

from app.translation.base import BatchMode, validate_batch_mode


class NoopTranslator:
    """Tarjima qilmaydi — matnni o'zgarishsiz qaytaradi."""

    name = "noop"
    enabled = False

    def __init__(self, batch_mode: BatchMode = "page") -> None:
        self.batch_mode = validate_batch_mode(batch_mode)

    def translate_pages(self, texts: list[str], target: str) -> list[str]:
        return list(texts)

    def healthcheck(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"NoopTranslator(batch_mode={self.batch_mode!r})"