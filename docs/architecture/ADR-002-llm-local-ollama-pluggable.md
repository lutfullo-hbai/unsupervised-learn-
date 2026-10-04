# ADR-002: LLM provayderi — Ollama (default), pluggable

## Context
M11'da clusterlarni tushuntirish uchun LLM kerak. Talablar: lokal ishlash (offline,
maxfiylik), o'rganish uchun qaror izlanishi, arzonlik. Lekin kelajakda bulutli
API kerak bo'lishi mumkin.

## Problem
LLM'ni qanday ulash: qat'iy Ollama'ga bog'lanishmi yoki almashtiriladigan
(layered) arxitektura qilishmi?

## Options Considered
| Variant | Qabul | Rad etish sababi |
|---------|-------|------------------|
| **A. Ollama default + `LLM_PROVIDER` orqali almashtiriladi** | ✅ Tanlangan | — |
| B. Faqat Ollama | | Kelajakda butun qatlamni qayta yozish kerak |
| C. Faqat OpenAI/cloud | | Offline ishlamaydi; maxfiylik buziladi |
| D. To'g'ridan-to'g'ri `requests` chaqiruvlari app/ ichida | | Qatlam aralashadi; testlash qiyin |

## Decision
`app/llm/base.py` (interfeys) + `app/llm/ollama_client.py` (adapter) +
`LLM_PROVIDER` config orqali tanlash.

**Default:** `ollama` — lokal, bepul, offline.
**Mavjud modellar:** `qwen3:4b` (tez), `qwen3:8b`, `nomic-embed-text`, `mxbai-embed-large`.

## Trade-offs
| Yoqotish | Yutish |
|----------|--------|
| Qo'shimcha qatlam — murakkablik | Provider almashadi — kelajakka yo'l |
| Ollama setup talab qiladi | Offline, bepul, maxfiy |
| Kichik model — sifat pasayishi | `qwen3:8b` bilan almashtiriladi |

## Consequences
- `app/nlp/` va `app/ml/` Ollama haqida **bilmaydi** (dependency inversion).
- Ollama ishlamasa → `OllamaUnavailableError` → HTTP **503**, ML natijasi saqlanadi.
- Timeout (120s) va retry (2) majburiy — `devops §58`, `§54`.
- M18'da `generate_json()` natijasi Pydantic bilan validatsiya qilinadi.

## Status
**Accepted** — M0 da qabul qilindi (2026-10-04)
