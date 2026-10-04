"""Tarjima provayderini konfiguratsiyaga qarab tanlash (ADR-003).

Nima uchun factory kerak (backend §48 «avoid unnecessary factories»):
tanlov **config ga bog'liq**, ya'ni qiymat kodda emas. Factory bu yagona
joy — pipeline'ning qolgan qismi provayder nomini umuman bilmaydi.

Yangi provayder qo'shishda yagona o'zgarish nuqtasi — `PROVIDERS`.
"""

from app.core.config import Settings, get_settings
from app.translation.base import Translator
from app.translation.exceptions import TranslationConfigError
from app.translation.google_translate import GoogleTranslateTranslator
from app.translation.noop import NoopTranslator

PROVIDERS: dict[str, str] = {
    "none": "NoopTranslator",
    "noop": "NoopTranslator",
    "google_translate": "GoogleTranslateTranslator",
}

NOOP_PROVIDERS = frozenset({"none", "noop"})


def get_translator(settings: Settings | None = None) -> Translator:
    """Config asosida translator qaytaradi.

    Args:
        settings: ishlatiladigan sozlamalar. Berilmasa `.env` dan o'qiladi.

    Returns:
        Tarjima o'chirilgan bo'lsa yoki provider `none`/`noop` bo'lsa
        `NoopTranslator`; aks holda tanlangan provayder.

    Raises:
        TranslationConfigError: provider nomi noma'lum, yoki
            `translation_enabled=true` bilan noop provider birga berilgan
            (bu ziddiyat jim qoldirilmaydi).
    """
    settings = settings or get_settings()

    provider = settings.translation_provider
    if provider not in PROVIDERS:
        allowed = " | ".join(sorted(PROVIDERS))
        raise TranslationConfigError(
            f"Noma'lum tarjima provideri: {provider!r}. Ruxsat etilgan: {allowed}."
        )

    if not settings.translation_enabled:
        return NoopTranslator(batch_mode=settings.translation_batch_by)

    if provider in NOOP_PROVIDERS:
        raise TranslationConfigError(
            "translation_enabled=true, lekin translation_provider=none. "
            "Birini o'zgartiring: provider='google_translate' yoki "
            "translation_enabled=false."
        )

    if provider == "google_translate":
        return GoogleTranslateTranslator(
            timeout=settings.google_translate_timeout,
            max_retries=settings.google_translate_max_retries,
            batch_mode=settings.translation_batch_by,
        )

    raise TranslationConfigError(f"Provider {provider!r} hali amalga oshirilmagan.")