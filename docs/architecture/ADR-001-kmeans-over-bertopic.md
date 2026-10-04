# ADR-001: Topic discovery uchun K-Means + TF-IDF (BERTopic emas)

## Context
Loyihaning maqsadi — **o'rganish**: unsupervised pipeline'ni noldan qurib, har bir komponent
sababini tushunish. Topik yechimi kerak edi. Bozor yechimi (BERTopic, Top2Vec) mavjud,
lekin ular ichida HDBSCAN/UMAP va transformer embedding kombinatsiyasini yashiradi —
qaysi qadam nima qilayotganini ko'rish imkoni yo'qoladi.

## Problem
Qanday topik yechim ishlatilsin? Kutubxonani chaqirish (BERTopic) yoki
asosiy algoritmlarni o'zing qurish (TF-IDF + K-Means)?

## Options Considered
| Variant | Qabul | Rad etish sababi |
|---------|-------|------------------|
| **A. TF-IDF + K-Means, o'zing qurish** | ✅ Tanlangan | — |
| B. BERTopic (kutubxona) | | Ichki bosqichlar yashiriladi; o'rganish maqsadiga zid |
| C. Top2Vec | | HDBSCAN va UMAP og'irligi; deterministik emas |
| D. Spherical K-Means (cosine) | | M6 da TF-IDF L2 normalizatsiya bilan bir xil; murakkablik orttiradi |
| E. LLM bilan embedding + klaster | | K-Means bilan aralashtirib chalkashlantiradi; alohida M9 da qoladi |

## Decision
**TF-IDF (M5) + K-Means (M6) + qo'lda top terms (M6)**, embeddings esa
**M9/M10 da alohida taqqoslash uchun**.

Sabab: pipeline'ning har bir bosqichini o'zlashtirish, natijani qo'lda tekshirish
(`top_terms` orqali), va keyin embeddings bilan qiyoslash mumkin bo'ladi.

## Trade-offs
| Yoqotish | Yutish |
|----------|--------|
| Topik sifati BERTopic'dan past bo'lishi mumkin | To'liq tushunish + nazorat |
| K-Means spherical klasterlashni yaxshi qilmaydi | Sodda, tez, deterministik |
| Topik sonini oldindan berish kerak | Avtomatik `k` tanlash M7 bilan qo'shiladi |

## Consequences
- M5, M6, M7 alohida, mustaqil testlanadigan modul bo'ladi.
- M9 embeddings — **murojaat qilinadigan qo'shimcha**, majburiy emas.
- `k` tanlash — sub'ktiv; M7 va M17 da hujjatlashtiriladi.
- If M9 ishonchi past bo'lsa, M5/M6 o'z-o'zidan ishlashda davom etadi.

## Status
**Accepted** — M0 da qabul qilindi (2026-10-04)
