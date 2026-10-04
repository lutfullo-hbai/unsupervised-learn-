"""Tarjima moduli umumiy kontrakti (ADR-003).

Bu modul **boshqa hech narsaga bog'liq emas** — shuning uchun uni
`app/core/config.py` ham xavfsiz import qilishi mumkin.

Chunk level tarjima **taqiqlangan** (ADR-003): kontekst yo'qoladi,
terminologiya buziladi va so'rovlar ko'payadi. `BatchMode` faqat ikki
qiymatni qabul qiladi; boshqasi `TranslationConfigError` beradi.
"""

from typing import Literal, Protocol, get_args, runtime_checkable

from app.translation.exceptions import TranslationConfigError

BatchMode = Literal["document", "page"]

VALID_BATCH_MODES: frozenset[str] = frozenset(get_args(BatchMode))


@runtime_checkable
class Translator(Protocol):
    """Tarjima provayderlari uchun umumiy interfeys.

    Yangi provayder qo'shish uchun shu protokol bajariladi — hech qanday
    mavjud fayl o'zgartirilmaydi (ADR-003: interface-first).
    """

    name: str
    enabled: bool

    def translate_pages(self, texts: list[str], target: str) -> list[str]:
        """Sahifa (yoki document) darajasida tarjima qiladi.

        Args:
            texts: tarjima qilinadigan matnlar ro'yxati (sahifalar yoki
                butun document).
            target: maqsad til kodi, masalan `"en"`.

        Returns:
            Kirish tartibida va o'zgarishsiz uzunlikdagi tarjimalar.

        Raises:
            TranslationUnavailableError: provider ishlamadi.
            TranslationResponseError: javob formati noto'g'ri.
        """
        ...

    def healthcheck(self) -> bool:
        """Provider mavjudligini tekshiradi.

        Returns:
            `True` agar provider javob bera olsa, `False` aks holda.
        """
        ...


def validate_batch_mode(mode: str) -> BatchMode:
    """`batch_by` qiymatini tekshiradi.

    Faqat `document` va `page` qabul qilinadi — **chunk level taqiqlangan**.

    Raises:
        TranslationConfigError: `mode` ruxsat etilgan qiymatlar orasida yo'q.
    """
    if mode not in VALID_BATCH_MODES:
        allowed = " | ".join(sorted(VALID_BATCH_MODES))
        raise TranslationConfigError(
            f"Taqiqlangan tarjima darajasi: {mode!r}. Ruxsat etilgan: {allowed}. "
            "Chunk level tarjima kontekstni buzadi (ADR-003)."
        )
    return mode  # type: ignore[return-value]