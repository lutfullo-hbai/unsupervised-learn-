"""M-T: tarjima moduli testlari.

Test nomlari xulqni tushuntiradi (qa §63):
`translate_pages_returns_input_unchanged_when_translation_disabled`.

Barcha tashqi so'rovlar `httpx.MockTransport` bilan almashtirilgan —
internet **kerak emas**, testlar deterministik. Haqiqiy integratsiya
testi alohida belgilangan va `-m "not integration"` bilan chiqariladi.
"""

import httpx
import pytest

from app.core.config import Settings
from app.translation import (
    GoogleTranslateTranslator,
    NoopTranslator,
    TranslationConfigError,
    TranslationError,
    TranslationResponseError,
    TranslationUnavailableError,
    Translator,
    get_translator,
    validate_batch_mode,
)
from app.translation.google_translate import MAX_TEXT_LENGTH


def gtx_response(text: str) -> httpx.Response:
    """`gtx` endpointining kutilgan javobini taqlid qiladi."""
    return httpx.Response(200, json=[[[text, "original"]]])


def client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- M-T.T1: batch mode validatsiyasi ---------------------------------


@pytest.mark.parametrize("mode", ["document", "page"])
def test_validate_batch_mode_accepts_document_and_page(mode):
    assert validate_batch_mode(mode) == mode


@pytest.mark.parametrize("mode", ["chunk", "CHUNK", "", "sentence", "paragraph"])
def test_validate_batch_mode_rejects_chunk_level_and_unknown_modes(mode):
    with pytest.raises(TranslationConfigError):
        validate_batch_mode(mode)


def test_translation_config_error_is_translation_error():
    assert issubclass(TranslationConfigError, TranslationError)
    assert issubclass(TranslationUnavailableError, TranslationError)
    assert issubclass(TranslationResponseError, TranslationError)


def test_chunk_level_is_rejected_by_both_providers():
    with pytest.raises(TranslationConfigError):
        NoopTranslator(batch_mode="chunk")
    with pytest.raises(TranslationConfigError):
        GoogleTranslateTranslator(batch_mode="chunk")


# --- M-T.T2: NoopTranslator --------------------------------------------


def test_translate_pages_returns_input_unchanged_when_translation_disabled():
    texts = ["salom", "Machine learning — bu nima?", ""]
    assert NoopTranslator().translate_pages(texts, "en") == texts


def test_noop_translate_pages_preserves_order_and_length():
    texts = [f"sahifa-{i}" for i in range(50)]
    result = NoopTranslator().translate_pages(texts, "en")
    assert result == texts
    assert len(result) == len(texts)


def test_noop_translate_pages_returns_new_list_not_same_object():
    texts = ["a", "b"]
    assert NoopTranslator().translate_pages(texts, "en") is not texts


def test_noop_translate_pages_handles_empty_input():
    assert NoopTranslator().translate_pages([], "en") == []


def test_noop_healthcheck_is_always_true():
    assert NoopTranslator().healthcheck() is True


def test_noop_reports_disabled():
    assert NoopTranslator().enabled is False
    assert NoopTranslator().name == "noop"


def test_noop_makes_no_network_call(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("NoopTranslator tarmoqqa ulanishga urinmasligi kerak")

    monkeypatch.setattr(httpx, "Client", explode)
    assert NoopTranslator().translate_pages(["salom"], "en") == ["salom"]


# --- M-T.T4: factory ---------------------------------------------------


def test_get_translator_returns_noop_when_translation_disabled():
    settings = Settings(translation_enabled=False, translation_provider="google_translate")
    assert isinstance(get_translator(settings), NoopTranslator)


def test_get_translator_returns_noop_by_default():
    assert isinstance(get_translator(Settings()), NoopTranslator)


def test_get_translator_returns_google_when_enabled():
    settings = Settings(translation_enabled=True, translation_provider="google_translate")
    translator = get_translator(settings)
    assert isinstance(translator, GoogleTranslateTranslator)
    assert translator.enabled is True


def test_get_translator_applies_config_to_google_provider():
    settings = Settings(
        translation_enabled=True,
        translation_provider="google_translate",
        translation_batch_by="document",
        google_translate_timeout=12.5,
        google_translate_max_retries=4,
    )
    translator = get_translator(settings)
    assert translator.timeout == 12.5
    assert translator.max_retries == 4
    assert translator.batch_mode == "document"


def test_get_translator_rejects_unknown_provider():
    """Factory o'z tekshiruvini bajarishi kerak — config validatsiyasidan
    kelib chiqib, alohida tekshiriladi."""
    settings = Settings()
    object.__setattr__(settings, "translation_provider", "deepl")
    object.__setattr__(settings, "translation_enabled", True)

    with pytest.raises(TranslationConfigError, match="deepl"):
        get_translator(settings)


def test_get_translator_rejects_enabled_with_noop_provider():
    settings = Settings(translation_enabled=True, translation_provider="none")
    with pytest.raises(TranslationConfigError):
        get_translator(settings)


def test_get_translator_result_satisfies_protocol():
    for settings in (
        Settings(),
        Settings(translation_enabled=True, translation_provider="google_translate"),
    ):
        assert isinstance(get_translator(settings), Translator)


def test_settings_rejects_chunk_level_batch_mode():
    with pytest.raises(Exception, match="chunk"):
        Settings(translation_batch_by="chunk")


def test_settings_rejects_unknown_provider():
    with pytest.raises(Exception):
        Settings(translation_provider="deepl")


def test_settings_rejects_negative_retries():
    with pytest.raises(Exception):
        Settings(google_translate_max_retries=-1)


# --- M-T.T3: GoogleTranslateTranslator ---------------------------------


def test_translate_pages_returns_translated_text_from_provider():
    translator = GoogleTranslateTranslator(
        client=client_with(lambda request: gtx_response("Hello"))
    )
    assert translator.translate_pages(["salom"], "en") == ["Hello"]


def test_translate_pages_sends_source_and_target_parameters():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return gtx_response("Hello")

    GoogleTranslateTranslator(client=client_with(handler)).translate_pages(["salom"], "en")

    assert seen["sl"] == "auto"
    assert seen["tl"] == "en"
    assert seen["dt"] == "t"
    assert seen["q"] == "salom"


def test_translate_pages_skips_request_for_blank_pages():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return gtx_response("Hello")

    result = GoogleTranslateTranslator(client=client_with(handler)).translate_pages(
        ["", "   ", "\n"], "en"
    )

    assert result == ["", "   ", "\n"]
    assert calls == []


def test_translate_pages_preserves_page_order():
    def handler(request: httpx.Request) -> httpx.Response:
        source = request.url.params["q"]
        return gtx_response(f"EN:{source}")

    texts = ["bir", "ikki", "uch", "to'rt"]
    result = GoogleTranslateTranslator(client=client_with(handler)).translate_pages(texts, "en")

    assert result == ["EN:bir", "EN:ikki", "EN:uch", "EN:to'rt"]


def test_translate_pages_joins_multiple_segments():
    payload = [[["Hello ", "x"], ["world", "y"]], None, "en"]
    client = client_with(lambda request: httpx.Response(200, json=payload))

    assert GoogleTranslateTranslator(client=client).translate_pages(["salom"], "en") == [
        "Hello world"
    ]


def test_translate_pages_returns_empty_list_for_empty_input():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("bo'sh ro'yxat uchun so'rov yuborilmasligi kerak")

    assert GoogleTranslateTranslator(client=client_with(handler)).translate_pages([], "en") == []


def test_translate_pages_splits_text_longer_than_endpoint_limit():
    """Endpoint 5000 belgidan uzun matnni qabul qilmaydi — bo'laklash kerak."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return gtx_response("X")

    long_text = "a" * (MAX_TEXT_LENGTH + 100)
    GoogleTranslateTranslator(client=client_with(handler)).translate_pages(
        [long_text], "en"
    )

    assert len(calls) == 2


def test_retryable_status_raises_unavailable_after_bounded_attempts():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503)

    translator = GoogleTranslateTranslator(
        client=client_with(handler), max_retries=2, backoff_base=0.0
    )

    with pytest.raises(TranslationUnavailableError):
        translator.translate_pages(["salom"], "en")

    assert len(attempts) == 3


def test_retryable_status_then_success_is_recovered():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429)
        return gtx_response("Hello")

    translator = GoogleTranslateTranslator(
        client=client_with(handler), max_retries=2, backoff_base=0.0
    )

    assert translator.translate_pages(["salom"], "en") == ["Hello"]
    assert len(attempts) == 2


def test_timeout_raises_unavailable_after_bounded_attempts():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectTimeout("timeout")

    translator = GoogleTranslateTranslator(
        client=client_with(handler), max_retries=1, backoff_base=0.0
    )

    with pytest.raises(TranslationUnavailableError):
        translator.translate_pages(["salom"], "en")

    assert len(attempts) == 2


def test_client_error_is_not_retried():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(400)

    translator = GoogleTranslateTranslator(
        client=client_with(handler), max_retries=2, backoff_base=0.0
    )

    with pytest.raises(TranslationResponseError, match="400"):
        translator.translate_pages(["salom"], "en")

    assert len(attempts) == 1


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "not-a-list",
        [None],
        [None],
        "x",
        [["ok"], "not-a-list"],
        [[]],
        [[123]],
        [["ok", None], None, "en"],
    ],
)
def test_malformed_provider_payload_raises_response_error(payload):
    client = client_with(lambda request: httpx.Response(200, json=payload))
    translator = GoogleTranslateTranslator(client=client, max_retries=0)

    with pytest.raises(TranslationResponseError):
        translator.translate_pages(["salom"], "en")


def test_non_json_response_raises_response_error():
    client = client_with(lambda request: httpx.Response(200, text="<html>blocked</html>"))
    translator = GoogleTranslateTranslator(client=client, max_retries=0)

    with pytest.raises(TranslationResponseError):
        translator.translate_pages(["salom"], "en")


def test_healthcheck_returns_true_when_provider_responds(monkeypatch):
    translator = GoogleTranslateTranslator(backoff_base=0.0)
    monkeypatch.setattr(
        translator, "_send", lambda client, text, target: gtx_response("a")
    )
    assert translator.healthcheck() is True


def test_healthcheck_returns_false_instead_of_raising(monkeypatch):
    translator = GoogleTranslateTranslator(backoff_base=0.0)

    def explode(client, text, target) -> httpx.Response:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(translator, "_send", explode)
    assert translator.healthcheck() is False


def test_healthcheck_does_not_leak_probe_client_to_next_call(monkeypatch):
    outer = client_with(lambda request: gtx_response("outer"))
    translator = GoogleTranslateTranslator(client=outer, backoff_base=0.0)
    monkeypatch.setattr(
        translator, "_send", lambda client, text, target: gtx_response("a")
    )

    assert translator.healthcheck() is True
    assert translator._client is outer

    monkeypatch.undo()
    assert translator.translate_pages(["salom"], "en") == ["outer"]
    outer.close()


def test_repr_does_not_leak_secrets():
    text = repr(GoogleTranslateTranslator())
    assert "key" not in text.lower()
    assert "password" not in text.lower()


# --- M-T o'zaro integratsiya ------------------------------------------


def test_pipeline_can_call_any_provider_without_branching():
    """M12 shu xususiyatga tayanadi: bitta kod, ikkala provayder."""
    texts = ["salom", "dunyo"]

    noop = get_translator(Settings(translation_enabled=False))
    assert noop.translate_pages(texts, "en") == texts

    google = get_translator(
        Settings(translation_enabled=True, translation_provider="google_translate"),
    )
    assert hasattr(google, "translate_pages")


@pytest.mark.integration
def test_real_google_translate_returns_english_text():
    """Real tarmoq testi — `pytest -m integration` bilan ishga tushadi."""
    translator = GoogleTranslateTranslator(timeout=20.0, max_retries=1)
    result = translator.translate_pages(["Tarjima moduli ishlayapti"], "en")
    translator._close()

    assert result[0]
    assert result[0] != "Tarjima moduli ishlayapti"