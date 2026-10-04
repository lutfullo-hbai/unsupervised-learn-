# ADR-005: Bitta FastAPI monolith + alohida Streamlit (monolith-first)

## Context
Loyiha o'rganish loyihasi. Docker M16 da kerak. Arxitektura tanlash kerak:
monolith, microservices, yoki SPA + API.

## Problem
Qanday arxitektura tanlanadi?

## Options Considered
| Variant | Qabul | Rad etish sababi |
|---------|-------|------------------|
| **A. FastAPI monolith + Streamlit (2 konteyner)** | ✅ Tanlangan | — |
| B. Microservices (LLM/ML alohida service) | | `system-architect §7` "Monolith First"; murakkablik oqlanmagan |
| C. React SPA + FastAPI | | Streamlit tezroq; React o'rganish maqsadiga kerak emas |
| D. Streamlit faqat (API siz) | | `system-architect §36` — Streamlit orqali ML pipeline qo'llab-quvvatlanmaydi |

## Decision
**FastAPI monolith** (barcha ML + LLM + translation uning ichida) + **Streamlit UI**
(alohida konteyner, faqat API chaqiradi).

## Trade-offs
| Yoqotish | Yutish |
|----------|--------|
| UI va API alohida — 2 ta jarayon | Frontend va backend chegarasi aniq |
| API kerak (Streamlit to'g'ridan-to'g'ri emas) | Testlash va qayta ishlatish oson |

## Consequences
- `app/` bitta dastur — microservice **yo'q**.
- Streamlit **majburiy ravishda** API orqali ishlaydi (`frontend §15`: "Do not invent endpoints").
- Model bir marta yuklanadi (`lifespan`) — har request'da emas.
- `app/main.py` — yagona kirish nuqtasi.

## Status
**Accepted** — M0 da qabul qilindi (2026-10-04)
