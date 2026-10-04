"""Loyiha sozlamalari — faqat `.env` dan o'qiladi (AGENTS.md §18).

Muhim qoida: **muhitga bog'liq qiymat kodga yozilmaydi.** Bu modul
`pydantic-settings` orqali `.env` ni o'qiydi va qiymatlarni validatsiya
qiladi — noto'g'ri konfiguratsiya ilk ishga tushirishda aniqlanadi,
ish vaqtida emas.

M-T kiritgan maydonlar `TRANSLATION_*` bilan boshlanadi. Ular
`TranslationConfigError` emas, `pydantic.ValidationError` beradi —
chunki bu **sozlash** vaqti xatosi, tarjima vaqti emas.

M2 qo'shdi `min_total_chars` — bu skanerlangan PDF chegarasi
va kodga yozilmaydi (AGENTS.md §18: konfiguratsiya faqat `.env` dan).
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = ".env"


class Settings(BaseSettings):
    """Barcha sozlamalar `.env` dan o'qiladi."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "PDF Analyzer"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    max_upload_mb: int = 25

    # --- PDF (M2) --------------------------------------------------------
    # Jami belgi chegarasi: ostida bo'lsa `ScannedPdfError`. chegara
    # juda past bo'lsa — oddiy PDF rad etiladi, juda yuqori bo'lsa —
    # skanerlangan PDF o'tib ketadi.
    min_total_chars: int = 100

    # --- LLM (M11) ------------------------------------------------------
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout: int = 120
    ollama_max_retries: int = 2
    ollama_num_ctx: int = 4096

    # --- Embeddings (M9) ------------------------------------------------
    use_multilingual: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_multilingual: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_batch_size: int = 16

    # --- Chunking (M4) --------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 200
    min_chunk_chars: int = 10

    # --- TF-IDF (M5) ----------------------------------------------------
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)

    # --- Clustering (M6/M7) ---------------------------------------------
    n_clusters: int | None = None
    kmeans_random_state: int = 42

    # --- Translation (M-T) — default OCHIQ ------------------------------
    translation_enabled: bool = False
    translation_provider: str = "none"
    translation_target: str = "en"
    translation_batch_by: str = "page"
    google_translate_timeout: float = 30.0
    google_translate_max_retries: int = 2

    @field_validator("translation_batch_by")
    @classmethod
    def _validate_batch_by(cls, value: str) -> str:
        """Chunk level taqiqlangan — `app.translation` importi ataylab
        shu yerda qilinadi, aks holda `app.core.config` va
        `app.translation.factory` o'zaro import tsikli hosil bo'lardi."""
        from app.translation.base import validate_batch_mode

        return validate_batch_mode(value)

    @field_validator("translation_provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        from app.translation.factory import PROVIDERS

        if value not in PROVIDERS:
            allowed = " | ".join(sorted(PROVIDERS))
            raise ValueError(f"provider={value!r} ruxsat etilmagan. Ruxsat: {allowed}")
        return value

    @field_validator("google_translate_max_retries")
    @classmethod
    def _validate_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_retries manfiy bo'lmaydi")
        if value > 5:
            raise ValueError("max_retries 5 dan katta bo'lmasligi kerak")
        return value

    @field_validator("translation_target")
    @classmethod
    def _validate_target(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or len(cleaned) > 8:
            raise ValueError(f"target til kodi noto'g'ri: {value!r}")
        return cleaned

    @field_validator("chunk_overlap")
    @classmethod
    def _validate_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_overlap manfiy bo'lmaydi")
        return value

    @field_validator("n_clusters")
    @classmethod
    def _validate_n_clusters(cls, value: int | None) -> int | None:
        if value is not None and value < 2:
            raise ValueError("n_clusters kamida 2 bo'lishi kerak (M6)")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """`.env` dan o'qilgan sozlamalarni **bir marta** yuklaydi va keshlaydi.

    Kesh — konfiguratsiya o'zgarmas (backend §35: caching faqat
    arxitektura qarori bilan; bunda `.env` o'zgarmas deb kelish oqilgan).
    Testlarda `get_settings.cache_clear()` chaqirish kerak.
    """
    return Settings()