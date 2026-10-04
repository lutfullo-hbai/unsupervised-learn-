# ADR-004: Python 3.12 + CPU-only PyTorch

## Context
M1 muhitini o'rnatish kerak edi. Sistematik Python **3.14.4** edi.
`torch` va `scikit-learn` uchun 3.14 da wheel yo'q.

## Problem
Qaysi Python va qanday torch build'ini ishlatish?

## Options Considered
| Variant | Qabul | Rad etish sababi |
|---------|-------|------------------|
| **A. Python 3.12 + CPU-only torch** | ✅ Tanlangan | — |
| B. Sistem Python 3.14 | | `torch`/`sklearn` wheel yo'q — o'rnatib bo'lmaydi |
| C. GPU torch (CUDA) | | ~2.5 GB; bu mashinada GPU yo'q; kerak emas |
| D. torch'siz (tf-idf + sklearn) | | Embeddings (M9) mumkin bo'lmaydi |

## Decision
`uv venv --python 3.12 .venv` (3.12.14) + `--index-url .../whl/cpu` (torch 2.14.1+cpu).

## Trade-offs
| Yoqotish | Yutish |
|----------|--------|
| GPU tezligi (bu holda kerak emas) | 2.5 GB → ~200 MB |
| 3.14 yangi xususiyatlari | Barcha wheel'lar mavjud |

## Consequences
- `requirements.txt` da torch **alohida** qo'yiladi (CPU index orqali).
- Docker image ham CPU-only (M16) — `docker-compose` da GPU kerak emas.
- `.env.example` da `OLLAMA_MODEL=qwen3:4b` (CPU'da tez ishlash uchun).

## Status
**Accepted** — M1 da qabul qilindi (2026-10-04)
