"""Google Translate provayderi — bepul, API kalitisiz.

Ishlatiladigan norasmiy `gtx` endpoints. Tanlov sabablari (ADR-003):
- API kaliti yo'q — lokal rivojlanish uchun to'g'ridan-to'g'ri ishlatiladi
- Bitta so'rovda bir sahifa yoki butun documentni tarjima qiladi

Cheklovlar:
- Endpoint **rasmiy emas** — o'zgarishi mumkin. Shu sababli javob formati
  tekshiriladi va noto'g'ri bo'lsa `TranslationResponseError` beriladi
- Rate limit mavjud, shuning uchun so'rovlar ketma-ket yuboriladi
- Har bir so'rovda matn **URL-kodlanadi**, shuning uchun `params` orqali
  yuboriladi (qo'lda URL yig'ish xavfsiz emas)
"""

import logging
import time

import httpx

from app.translation.base import BatchMode, validate_batch_mode
from app.translation.exceptions import (
    TranslationResponseError,
    TranslationUnavailableError,
)

logger = logging.getLogger(__name__)

GOOGLE_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

BACKOFF_BASE_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 8.0
MAX_TEXT_LENGTH = 4800


class GoogleTranslateTranslator:
    """Google Translate orqali sahifa/document darajasida tarjima."""

    name = "google_translate"
    enabled = True

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 2,
        batch_mode: BatchMode = "page",
        client: httpx.Client | None = None,
        backoff_base: float = BACKOFF_BASE_SECONDS,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_mode = validate_batch_mode(batch_mode)
        self.backoff_base = backoff_base
        self._client = client
        self._owns_client = client is None

    # --- so'rov yuborish -------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def _build_params(self, text: str, target: str) -> dict[str, str]:
        return {
            "client": "gtx",
            "sl": "auto",
            "tl": target,
            "dt": "t",
            "q": text,
        }

    def _send(
        self, client: httpx.Client, text: str, target: str
    ) -> httpx.Response:
        return client.get(
            GOOGLE_TRANSLATE_URL,
            params=self._build_params(text, target),
            timeout=self.timeout,
        )

    def _request_once(self, text: str, target: str) -> httpx.Response:
        return self._send(self._get_client(), text, target)

    def _request_with_retry(self, text: str, target: str) -> httpx.Response:
        """Cheklangan retry: exponential backoff, hech qachon infinite loop yo'q.

        Qayta uriniladigan holatlar: timeout, tarmoq xatosi, 408/425/429/5xx.
        4xx (429 dan tashqari) qayta urinishga **ruxsat berilmaydi** —
        so'rovning o'zi noto'g'ri.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._request_once(text, target)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning(
                    "Tarjima so'rovi tarmoq xatosi berdi (attempt %s/%s): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    type(exc).__name__,
                )
            else:
                if response.status_code in RETRYABLE_STATUS_CODES:
                    last_error = TranslationUnavailableError(
                        f"Provider vaqtinchalik javob berdi: HTTP {response.status_code}"
                    )
                    logger.warning(
                        "Tarjima so'rovi retryable javob berdi (attempt %s/%s): HTTP %s",
                        attempt + 1,
                        self.max_retries + 1,
                        response.status_code,
                    )
                elif response.is_error:
                    raise TranslationResponseError(
                        f"Provider so'rovni rad etdi: HTTP {response.status_code}"
                    )
                else:
                    return response

            if attempt < self.max_retries:
                delay = min(self.backoff_base * (2**attempt), BACKOFF_MAX_SECONDS)
                time.sleep(delay)

        raise TranslationUnavailableError(
            f"Tarjima {self.max_retries + 1} urinishdan keyin bajarilmadi: {last_error}"
        ) from last_error

    # --- javobni o'qish --------------------------------------------------

    @staticmethod
    def _parse_translation(payload: object) -> str:
        """`gtx` javobining birinchi segmentlarini bitta matnga bog'laydi.

        Bekkutilgan tuzilma: `[[["tarjima", "original", ...], ...], ...]`
        """
        if not isinstance(payload, list) or not payload:
            raise TranslationResponseError(
                f"Provider javobi ro'yxat emas yoki bo'sh: {type(payload).__name__}"
            )

        segments = payload[0]
        if not isinstance(segments, list):
            raise TranslationResponseError(
                f"Provider javobida segmentlar yo'q: {type(segments).__name__}"
            )

        parts: list[str] = []
        for segment in segments:
            if not isinstance(segment, list) or not segment:
                raise TranslationResponseError(
                    f"Provider javobida noto'g'ri segment: {segment!r:.80}"
                )
            head = segment[0]
            if not isinstance(head, str):
                raise TranslationResponseError(
                    f"Tarjima matni string emas: {type(head).__name__}"
                )
            parts.append(head)

        if not parts:
            raise TranslationResponseError(
                "Provider bo'sh tarjima qaytardi (segmentlar yo'q)"
            )

        return "".join(parts)

    def _translate_text(self, text: str, target: str) -> str:
        if not text.strip():
            return text

        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(
                "Matn %s belgidan uzun, bo'laklarga ajratiladi (sahifa chegarasi)",
                len(text),
            )
            chunks = [
                text[i : i + MAX_TEXT_LENGTH]
                for i in range(0, len(text), MAX_TEXT_LENGTH)
            ]
            return "".join(self._translate_text(chunk, target) for chunk in chunks)

        response = self._request_with_retry(text, target)
        try:
            payload = response.json()
        except ValueError as exc:
            raise TranslationResponseError(
                f"Provider javobi JSON emas: {exc}"
            ) from exc
        return self._parse_translation(payload)

    # --- Translator protokoli -------------------------------------------

    def translate_pages(self, texts: list[str], target: str) -> list[str]:
        validate_batch_mode(self.batch_mode)
        if not texts:
            return []
        return [self._translate_text(text, target) for text in texts]

    def healthcheck(self) -> bool:
        """Bitta minimal so'rov bilan mavjudlikni tekshiradi.

        Alohida qisqa muddatli klient ishlatiladi va `self._client` **ga
        tegilmaydi** — keyingi so'rovlar o'z klientidan foydalanadi.

        Xato bo'lsa `False` qaytaradi (xato otmaydi) — `/health` endpointi
        boshqa xatolardan farqlanishi uchun (devops §64: "Do not mark a
        service healthy when it cannot actually perform its work", lekin
        healthcheck o'zi muvaffaqiyatsiz bo'lsa exception emas).
        """
        try:
            with httpx.Client(timeout=min(self.timeout, 5.0)) as probe:
                payload = self._send(probe, "a", "en").json()
            self._parse_translation(payload)
        except Exception:  # noqa: BLE001 - healthcheck hech qachon xato otmaydi
            logger.info("Tarjima healthcheck muvaffaqiyatsiz", exc_info=True)
            return False
        return True

    def __enter__(self) -> "GoogleTranslateTranslator":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._close()

    def __repr__(self) -> str:
        return (
            f"GoogleTranslateTranslator(batch_mode={self.batch_mode!r}, "
            f"timeout={self.timeout}, max_retries={self.max_retries})"
        )