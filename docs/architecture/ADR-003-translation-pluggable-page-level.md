# ADR-003: Translation — pluggable, default ochiq, faqat page/document level

## Context
Foydalanuvchi turli tilli PDF'larni qo'llab-quvvatlamoqchi. Leki mavjud
pipeline ingliz tiliga optimallashtirilgan. Google Translate bepul, lekin
tashqi servis (maxfiylik + rate limit + offline yo'q).

## Problem
Tarjimani qayerda va qanday darajada qilish?

## Options Considered
| Variant | Qabul | Rad etish sababi |
|---------|-------|------------------|
| **A. Pluggable, default `Noop`, faqat `document`/`page`** | ✅ Tanlangan | — |
| B. Default `google_translate` `true` | | Maxfiylik buziladi, offline ishlamaydi, majburiy emas |
| C. **Chunk level** tarjima | | ❌ **Kontekst yo'qoladi**, terminologiya buziladi, so'rovlar ko'payadi |
| D. Faqat bitta provider (Google) | | Kelajakda qo'shish uchun qayta yozish kerak |
| E. Ollama orqali tarjima (local LLM) | | Sifat sekin; M11'dan keyin qo'shish mumkin |

## Decision
**A.** `app/translation/base.py` (Protocol) + `noop.py` (default) +
`google_translate.py` (ixtiyoriy) + `factory.py`.
Faqat `TRANSLATION_BATCH_BY ∈ {document, page}`. Chunk level **taqiqlangan**.

## Trade-offs
| Yoqotish | Yutish |
|----------|--------|
| M5/M9/M12/M13 bilan bog'liqlik | Aralash tilli PDF qo'llab-quvvatlanadi |
| Qo'shimcha konfiguratsiya | Default xavfsiz va offline |
| Google rate limit | Retry + 503 bilan boshqariladi |

## Consequences
- `stop_words` config'dan keladi: tarjima yo'q → `None` (ko'p tilli), bor → `"english"`.
- Tarjima **pre-chunk** (chunk'dan oldin).
- LLM maydonlari `None` bo'lishi mumkin, lekin `topics` **hech qachon bo'sh qolmaydi**.
- Xato → `TranslationUnavailableError` → **503** (`app/main.py`).

## Status
**Accepted** — M0 da qabul qilindi (2026-10-04)
