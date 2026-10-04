# unsupervised-learn-
# PDF Analyzer — Unsupervised Learning loyihasi

> **Bu README — loyihaning "ish daftari".** Har safar o'tirib ishlaganda shu faylga qarab: *hozir qaysi milestone'dasan, qaysi faylda ishlaysan, nima qilasan, qachon "tayyor" deb hisoblaysan.*
>
> **Turi:** End-to-End ML + NLP + LLM ilova · **Asosiy yo'nalish:** Unsupervised Learning
> **Til:** Python 3.12 · **LLM:** Ollama (pluggable) · **Backend:** FastAPI · **Frontend:** Streamlit
> **Translation:** pluggable (config), **default OCHIQ** (local-first) · **Paper:** [arxitektura](docs/architecture.md) · [I/O contract](docs/io_contract.md)

---

## Mundarija

1. [Loyiha nima qiladi](#1-loyiha-nima-qiladi)
2. [Har kuni qanday ishlash (Ish tartibi)](#2-har-kuni-qanday-ishlash)
3. [Arxitektura](#3-arxitektura)
4. [Papka tuzilmasi — qaysi fayl qachon yaratiladi](#4-papka-tuzilmasi)
5. [Texnologiyalar](#5-texnologiyalar)
6. [Loyiha tamoyillari](#6-loyiha-tamoyillari)
7. [Milestone'lar M0–M18 + M-T (task breakdown bilan)](#7-milestonelar)
8. [Baholash (Evaluation) protokoli](#8-baholash-protokoli)
9. [Edge case'lar jadvali](#9-edge-caselar)
10. [Git va commit qoidalari](#10-git-qoidalari)
11. [Birinchi versiyada QILINMAYDIGAN narsalar](#11-non-goals)
12. [Kelajak yo'l xaritasi](#12-kelajak)
13. [Progress tracker](#13-progress-tracker)
14. [Ish jurnali shabloni](#14-ish-jurnali)
15. [Lug'at (Glossary)](#15-lugat)

---

# 1. Loyiha nima qiladi

Foydalanuvchi uzun PDF yuklaydi (ilmiy maqola, texnik hujjat, hisobot, o'quv material, kitob). Tizim:

| # | Natija | Kim chiqaradi |
|---|--------|---------------|
| 1 | Asosiy mavzu | ML + LLM |
| 2 | Topilgan mavzular (topics) | **Unsupervised ML** (clustering) |
| 3 | Mavzular bo'yicha guruhlash | **Unsupervised ML** |
| 4 | Muhim bo'limlar | ML (markazga yaqin chunk'lar) |
| 5 | Asosiy fikrlar (key points) | LLM |
| 6 | Umumiy xulosa | LLM |
| 7 | Odam tushunadigan tushuntirish | LLM |

**Eng muhim talab:** mavzularni topish **oldindan belgilangan label'larsiz** (unsupervised) bo'lishi kerak. LLM mavzularni *topmaydi* — u faqat ML topgan tuzilmani *tushuntiradi*.

```text
✅ TO'G'RI:   PDF → Unsupervised ML → Topilgan tuzilma → LLM tushuntiradi
❌ NOTO'G'RI: PDF → LLM → "mana bir nechta mavzular"
```

## 1.1. Turli tilli PDF'lar (Translation moduli — M-T)

Tarjima **majburiy emas**, **config orqali boshqariladi** (pluggable provider). Tamoyil: **local-first by default + cloud-optional**.

| Rejim | Sozlama | Pipeline | Qachon |
|-------|---------|----------|--------|
| **A. Asl matn** (default) | `TRANSLATION_ENABLED=false`<br>`USE_MULTILINGUAL=true` | Ko'p tilli embedding + `stop_words=None` | Ko'pchilik hollatlar |
| **B. Inglizga tarjima** | `TRANSLATION_ENABLED=true`<br>`TRANSLATION_PROVIDER=google_translate` | Monolingual embedding + `stop_words="english"` | Aralash tilli PDF, faqat **zarur bo'lsa** |

**Qat'iy qoidalar:**

1. **Hech qachon chunk darajasida tarjima qilinmaydi** — faqat `document` yoki `page`.
2. Tarjima **pre-chunk**: sahifa darajasida bajariladi, keyin normal chunking.
3. Default — `NoopTranslator` (passthrough, **tashqi so'rov umuman yuborilmaydi**).
4. Xato → `TranslationError` / `TranslationUnavailableError` (→ HTTP 503).
5. **Interface-first** (`base.py` + Factory) — yangi provider qo'shish oson.

**Nima uchun default A?** Aralash tilli PDF'da cluster'lar **til bo'yicha** bo'linishi mumkin. Multilingual embedding bitta vektor fazosida solishtiradi. Tarjima esa monolingual pipeline'ni toza qiladi, lekin **xarjat + maxfiylik** talab qiladi. Qaysi biri yaxshi — **M17'da o'lchanadi**. Hozir default — arzon va halol yo'l.

> **Test/tajribada:** Google Translate free tier yetarli. Production'da o'chirish yoki `libre_translate` (self-host) ga o'tish mumkin.

---

# 2. Har kuni qanday ishlash

## 2.1. Sessiya boshlash (5 daqiqa)

1. Shu README'ning **[13. Progress tracker](#13-progress-tracker)** bo'limini och — qaysi task'da ekaningni ko'r.
2. O'sha task ning **7.x** bo'limidagi kartasiga o't — **birinchi belgilanmagan `[ ]` taskni** (`Mx.Tn`) top.
3. `git status` va `git log --oneline -5` — oxirgi holatni eslab ol.
4. Virtual environment'ni yoq:
   ```bash
   source .venv/bin/activate          # Linux/macOS
   .venv\Scripts\activate             # Windows
   ```

## 2.2. Har bir task uchun majburiy workflow

Har bir task (trivial bo'lmagan) **shu 8 bosqichni** ketma-ket bajaradi. Bosqichni o'tkazib yuborma:

```text
1. Understand  → Talabni aniqlash: nima qilinadi, qanday ishlaydi,
                 input/output nima, qanday xatolar bo'lishi mumkin
2. Inspect     → Mavjud kodni o'qish: papka tuzilmasi, mavjud modullar,
                 sxemalar, testlar, config
3. Plan        → Implementation rejasi: qaysi fayl, qanday o'zgarish,
                 qanday test, qanday xavflar
4. Implement   → Kod yozish (minimal, aniq, mavjud koddan foydalanish)
5. Test        → Test yozish va yurgizish
6. Review      → O'zgartirishni tekshirish: nomlar, tuzilma, takror, izolyatsiya
7. Validate    → Taskning "Qanday tekshirasan" mezonini bajarish
8. Report      → Qisqa hisobot: nima qilindi, nima qoldi, keyingi qadam
```

**Qoidasi:** `Plan` tugamaguncha kod yozma. `Validate` o'tmaguncha task tugagan hisoblanma.

> Bu oqim `opencode/AGENTS.md` §34 va `opencode/backend-engineer.md` §4 bilan mos.
> `opencode/frontend-engineer.md` §4 ham **xuddi shu 8 bosqichni** talab qiladi.

## 2.2.1. Rolga xos workflow'lar (majburiy)

Har agent o'z hujjatidagi oqimini bajaradi. Bular bir-birini almashtirmaydi:

| Rol | Hujjat | Workflow |
|-----|--------|----------|
| Backend | `backend-engineer.md` §4 | `Understand → Inspect → Plan → Implement → Test → Review → Validate → Report` |
| Frontend | `frontend-engineer.md` §4 | `Understand → Inspect → Plan → Implement → Test → Review → Validate → Report` |
| **QA** | `qa-engineer.md` §5 | `Understand Requirements → Identify Acceptance Criteria → Risk Analysis → Test Plan → Test Design → Implementation → Execute → Defect Analysis → Regression → Quality Decision → Report` (**11 bosqich**) |
| **Security** | `security-engineer.md` §5 | `Identify → Model → Prevent → Implement → Test → Monitor → Respond → Improve` (**8 bosqich**) |
| **System Architect** | `system-architect.md` §3 | `Business Req → Functional Req → Non-Functional Req → Constraints → System Boundaries → Component Arch → Data Arch → Communication Arch → Infra Arch → Implementation` |
| **System Architect** | `system-architect.md` §46 | `Inspect → Understand → Measure impact → Design alternative → Compare trade-offs → Document decision → Get approval → Implement → Validate` |

> **"Do NOT start with technologies."** (`system-architect.md` §3) — texnologiyadan emas, **talabdan** boshlanadi.

## 2.2.2. Quality Gates (AGENTS.md §44)

Har feature ketma-ket o'tishi kerak. Kritik gate'da to'xtaydi.

```text
Requirement Gate → Architecture Gate → Implementation Gate → Test Gate
→ Code Quality Gate → Security Gate → Operational Gate → Documentation Gate → Release
```

| Gate | Qachon o'tadi | Milestone |
|------|---------------|-----------|
| Requirement | Scope, I/O, acceptance criteria aniq | M0 |
| Architecture | Arxitektura + ADR | M0 / M18 |
| Implementation | Kod yozildi | M2–M14 |
| Test | Testlar yozildi va o'tdi | M15 |
| Code Quality | Naming, tuzilma, takror yo'q | M18 |
| Security | Xavflar ko'rib chiqildi | M13 / M16 / M18 |
| Operational | Docker, health, logging | M16 |
| Documentation | README + `docs/` yangilandi | M18 |

## 2.2.3. "No Fake Completion" (AGENTS.md §39) — ENG MUHIM

**Tasdiqlanmagan narsani `Tasdiqlandi`/`Ishlayapti`/`Test o'tdi` deb yozish TAQIQLANGAN.**

- Tugatildi deb aytganda → **aniq dalil** bo'lishi shart (chiqarilgan natija, o'tgan test, screenshot).
- Dalil yo'q bo'lsa → **`NOT VERIFIED`** deb yoz va **nega** deb tushuntir.
- "Ishlayapti shekilli" → bu dalil **emas**.

> Ayniqsa: `pytest` ishlatilmagan bo'lsa "testlar o'tdi" **yozilmaydi**. Faqat `import` tekshirilgan bo'lsa — "import tekshirildi" deb yoziladi.

## 2.2.4. Change management (AGENTS.md §31)

Mavjud arxitekturani o'zgartirishdan **OLDIN** 7 ta savolga javob ber:

1. Dependency'lar ta'sirlanadimi? 2. Behavior o'zgaradimi? 3. API buziladimi?
4. Qaysi ma'lumot ta'sirlanadi? 5. Qaysi testlar o'zgaradi?
6. Migration kerakmi? 7. Rollback mumkinmi?

**Breaking change** bo'lsa — ochiq belgilanadi, migration strategiyasi va versiyalash yoziladi (§32).

## 2.2.5. Escalation (qachon to'xtab, so'rash kerak)

**To'xtab, foydalanuvchiga so'raganda:**

| Holat | Nima qilish |
|-------|-------------|
| Talab noaniq va natijani o'zgartiradi | To'xta, savol ber (§4) |
| Arxitektura mojjarosi | To'xta, muhokama qil (§35, §43) |
| Yangi infratuzilma / taqsimlangan tizim kerak | To'xta, arxitektura review (§36) |
| Zaiflik kritik darajada | To'xta, xavfsizlikka xabar ber (security §12) |
| Migratsiya xavfi katta | To'xta, ma'lumot arxitekturasiga murojaat (data §71) |
| Hajm/performance talabi bajarilmaydi | To'xta, arxitektura bo'yicha qaror (system-arch §46) |

> **Hech qachon jim qolib arxitektura o'zgartirma.** (`AGENTS.md` §36, §43)

## 2.2.6. Manba ustuvorligi (AGENTS.md §47)

Konflikt bo'lganda shu tartib:

```text
1. Loyiha talablari (README §6)
2. Global engineering constitution (opencode/AGENTS.md)
3. Arxitektura qarorlari (docs/architecture/ADR-*.md)
4. Mavjud arxitektura (docs/architecture.md)
5. API contract'lar (docs/io_contract.md)
6. Agent rol qoidalari (opencode/*.md)
7. Implementation details
```

> Hujjat **tekshirilgan haqiqiyatga** zid bo'lsa — **farqni hisobotda ko'rsatish majburiy** (§47).

> `opencode/` — **ichki agent/workflow qoidalari**, loyiha kodi emas. `.gitignore`da
> `opencode/` bilan chiqarilgan, shuning uchun **klonlangan repo'da bu fayllar yo'q**.
> Yuqoridagi konflikt tartibi mahalliy ish muhitidagi `opencode/` fayllariga ishora qiladi.

## 2.2.7. Definition of Done (AGENTS.md §40) — har task uchun

```text
[ ] Talab tushunildi
[ ] Arxitektura ko'rib chiqildi
[ ] Implementatsiya bajarildi
[ ] Loyiha standartlariga mos
[ ] Testlar qo'shildi yoki yangilandi
[ ] Testlar YURGIZILDI (yozish ≠ ishga tushish)
[ ] Xavfsizlik ta'siri ko'rib chiqildi
[ ] API contract' yangilandi (agar kerak)
[ ] DB migration yaratildi (agar kerak)
[ ] Dokumentatsiya yangilandi (agar kerak)
[ ] unrelated o'zgarishlar kiritilmadi
[ ] Yakuniy natija tekshirildi
```

> **Rolga xos kengaytmalar** (§1: agent hujjatlari kengaytirishi mumkin, **zid bo'lmaydi**):
> - Frontend `frontend-engineer.md` §67 — **21 ta** checkbox
> - DevOps `devops-engineer.md` §77 — **17 ta** checkbox
> - QA `qa-engineer.md` §67 — **18 ta** checkbox
> - Security `security-engineer.md` §81 — **14 ta** checkbox

## 2.2.8. Mandatory Completion Report (AGENTS.md §51)

Har muhim task oxirida **majburiy** hisobot:

```text
## Task              — nima vazifa berilgan
## Completed         — nima qilindi
## Files Changed     — qaysi fayllar
## Architecture Impact — yo'q / tavsif
## Tests             — qaysi testlar, N/M o'tdi
## Validation        — qanday tekshirildi
## Risks             — qanday xavf qoldi
## Assumptions       — nima taxmin qilindi
## Not Verified      — nimani tekshira olmadim
## Remaining Work    — keyingi qadam
```

> Kengaytmalar: frontend `§71` — **15** sarlavha; devops `§81` — **20** sarlavha; QA `§71` — `QA Report` shabloni; security `§85` — `Security Review` shabloni.
> **Emoji ishlatma** — faqat strukturali ma'lumot (§41).

## 2.3. Har bir ML tushunchasi uchun 10-qadam (MAJBURIY)

ML komponentlari uchun quyidagi 10 qadam **ketma-ket** bajariladi:

```text
 1. Concept        → bu nima? (1–2 jumla bilan o'z so'zing bilan yoz)
 2. Intuition      → nega kerak? qaysi muammoni hal qiladi?
 3. Small example  → 3–5 ta qisqa gap/nuqta bilan qo'lda hisobla
 4. Math           → formulani qog'ozga yoz, bir qiymatni qo'lda chiqar
 5. Manual code    → faqat NumPy/Python bilan o'zing yoz (notebooks/)
 6. Library        → scikit-learn/kutubxona versiyasi, natijani 5-qadam bilan solishtir
 7. Real PDF       → haqiqiy PDF'da ishlat (data/raw/)
 8. Evaluate       → natija ma'noli-mi? (8-bo'limga qara)
 9. Integrate      → app/ ichiga toza modul sifatida ko'chir + test yoz
10. Document       → "Nimani o'rgandim" va "Cheklovlar" ni yoz (docs/ yoki notebook oxiriga)
```

**Taqiqlangan sikl:** `kodni copy qil → run qil → "ishlayapti" de`. Maqsad — PDF analyzer yaratish emas, balki **har bir komponent nega mavjud, nima qiladi, qanday ishlaydi va qanday baholanishini tushunish.**

## 2.4. Qayerda nima yoziladi (qoida)

| Nima qilyapsan | Qayerda |
|----------------|---------|
| Tushunish, tajriba, vizualizatsiya | `notebooks/NN_nom.ipynb` |
| Yakuniy, qayta ishlatiladigan kod | `app/...` |
| Boshqaruvchi skriptlar | `scripts/...` |
| Tekshiruv | `tests/...` |
| Namuna PDF'lar | `data/raw/` (`.gitignore` da — katta PDF'lar commit qilinmaydi) |
| Oraliq natijalar (keshlangan matn, chunk'lar) | `data/processed/` |
| Hisobotlar, rasmlar, baholash natijalari | `docs/` + `docs/figures/` |

**Tartib:** avval notebook'da tajriba → ishlagach `app/` ga ko'chir → test yoz → commit.

## 2.5. Sessiya tugatish (5 daqiqa)

1. Tracker'da tugagan `[ ]` larni `[x]` qil (`Mx.Tn` ko'rinishida).
2. **Majburiy hisobot** yoz (2.2.8 — 10 sarlavha).
3. [14. Ish jurnali](#14-ish-jurnali) ga 3 qator yoz: *nima qildim / nimani tushunmadim / ertaga nima.*
4. `git add -A && git commit -m "M5: add TfidfVectorizer wrapper + tests"`.

> **Commit faqat foydalanuvchi aniq so'raganda** qilinadi. Har `[x]` — bu tracker yangilanishi, avtomatik commit emas.
> **Commit atomik bo'lishi shart** — feature + unrelated refactor + formatlashni aralashtirma (`AGENTS.md` §29).

## 2.6. "Tushunmasam nima qilaman" qoidasi

Agar 5-qadam (manual code) ni yoza olmasang — demak 3–4 qadamni (kichik misol + math) tushunmading. Orqaga qayt. **Manual implementatsiyasiz sklearn'ga o'tma.** (Istismo: PCA/SVD kabi murakkab narsalarda faqat intuitsiya + kichik misol yetarli.)

---

# 3. Arxitektura

```text
                         PDF
                          │
                          ▼
                 ┌─────────────────┐
                 │ PDF Text Extract │   M2  (PyMuPDF)
                 └────────┬────────┘
                          ▼
                ┌───────────────────────┐
                │ [OPTIONAL] Translate   │  M-T (document/page level)
                │  noop | google_translate│
                └────────┬──────────────┘
                          ▼
                 ┌─────────────────┐
                 │ Text Processing │   M3
                 └────────┬────────┘
                          ▼
                       Chunking           M4
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
          TF-IDF (M5)           Embeddings (M9)
              │                       │
              ▼                       ▼
         K-Means (M6)            Similarity
              │                       │
              └───────────┬───────────┘
                          ▼
                 Topic Discovery          M10
                          ▼
                   LLM (M11)        pluggable: ollama | openai | ...
                          ▼
                 Final Explanation        M12
                          ▼
                       FastAPI            M13
                          ▼
                    Streamlit UI          M14
```

**Qatlamlar (alohida saqlanadi, bir-biriga aralashmaydi):**

| Qatlam | Papka | Javobgarlik | Nimani BILMASLIGI kerak |
|--------|-------|-------------|--------------------------|
| PDF | `app/services/pdf_extractor.py` | PDF → sahifa matni | ML, LLM, API, Translation haqida |
| **Translation** | `app/translation/` | Pluggable tarjima (noop / google_translate), document/page level | ML, API, pipeline ichki tafsilotlari |
| NLP | `app/nlp/` | tozalash, chunking | ML, LLM, API, Translation haqida |
| ML | `app/ml/` | vektorlash, clustering, baholash | PDF, LLM, API, Translation haqida |
| LLM | `app/llm/` | LLM aloqa (pluggable), prompt'lar | ML algoritmlari haqida |
| Service | `app/services/` | yuqoridagilarni bog'lovchi pipeline | HTTP haqida |
| API | `app/api/` | HTTP, validatsiya, xatolar | ML/Translation ichki tafsilotlari haqida |

**Translation arxitekturasi (Strategy + Factory, interface-first):**

```text
app/translation/
├── __init__.py           public API
├── base.py               TranslatorProtocol / BaseTranslator  ← INTERFACE
├── factory.py            get_translator()  ← config asosida
├── noop.py               NoopTranslator  (DEFAULT — passthrough)
├── google_translate.py   GoogleTranslateTranslator  (optional)
└── exceptions.py         TranslationError, TranslationUnavailableError
```

> Yangi provider qo'shish uchun faqat `base.py` ni meris qilib yangi fayl yozish + `factory.py` ga bir qator qo'shish kifoya. `base.py` o'zgarmaydi.

---

# 4. Papka tuzilmasi

```text
pdf-analyzer/
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # M13  FastAPI app
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_documents.py    # M13  /documents/*
│   │   └── routes_health.py       # M13  /health
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # M1   Settings (pydantic-settings) — hammasi .env dan
│   │   ├── logging.py             # M18  logging
│   │   └── exceptions.py          # M2   maxsus xatolar
│   ├── translation/               # M-T  PLUGGABLE TRANSLATION MODULE
│   │   ├── __init__.py
│   │   ├── base.py                # TranslatorProtocol / BaseTranslator (interface)
│   │   ├── factory.py             # get_translator()  — config asosida
│   │   ├── noop.py                # NoopTranslator (DEFAULT — lokal passthrough)
│   │   ├── google_translate.py    # GoogleTranslateTranslator (optional, test uchun)
│   │   └── exceptions.py          # TranslationError, TranslationUnavailableError
│   │
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # M3
│   │   └── chunking.py            # M4
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── vectorizers.py         # M5, M9  TF-IDF + embeddings
│   │   ├── clustering.py          # M6
│   │   ├── evaluation.py          # M7
│   │   ├── visualization.py       # M8
│   │   └── topics.py              # M10
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_client.py       # M11
│   │   └── prompts.py             # M11
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py       # M2
│   │   └── analysis_service.py    # M12 to'liq pipeline
│   └── schemas/
│       ├── __init__.py
│       ├── document.py            # M2/M4 ichki dataclass/Pydantic
│       └── analysis.py            # M13 API javob sxemalari
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/                  # M15 kichik test PDF'lar
│   ├── test_pdf_extractor.py
│   ├── test_preprocessing.py
│   ├── test_chunking.py
│   ├── test_vectorizers.py
│   ├── test_clustering.py
│   ├── test_topics.py
│   ├── test_ollama_client.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── notebooks/
│   ├── 01_pdf_extraction.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_chunking.ipynb
│   ├── 04_tfidf_manual_and_sklearn.ipynb
│   ├── 05_kmeans_manual_and_sklearn.ipynb
│   ├── 06_cluster_evaluation.ipynb
│   ├── 07_pca_visualization.ipynb
│   ├── 08_embeddings.ipynb
│   ├── 09_topic_discovery.ipynb
│   └── 10_final_evaluation.ipynb
│
├── data/
│   ├── raw/                       # namuna PDF'lar (git'iga katta fayl qo'shma)
│   │   └── real/                  # real korpus — .gitignore'da (10 kitob + 5 OA PDF)
│   ├── processed/                 # keshlangan chunk'lar, vektorlar
│   └── manifest.json              # haqiqiy korpus: manba, litsenziya, SHA-256
│
├── scripts/
│   ├── fetch_real_documents.py    # real korpusni yuklash (Gutenberg/arXiv/PLOS)
│   ├── make_real_pdfs.py          # Gutenberg matni -> PDF (pymupdf Story)
│   └── make_sample_pdfs.py        # M0  namuna PDF'lar yaratish (PyMuPDF)
│
├── docs/
│   ├── architecture.md            # M0   arxitektura brief (20 bo'lim)
│   ├── io_contract.md             # M0   Input/Output contract + baholash mezonlari
│   ├── evaluation_report.md       # M17
│   ├── limitations.md             # M2–M3 (L-01..L-15), M17 da kengaytiriladi
│   ├── learning_notes.md          # har milestone'da "nimani o'rgandim"
│   └── figures/                   # M8/M17 grafiklar
│
├── frontend/
│   └── streamlit_app.py           # M14
│
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile                     # M16
├── docker-compose.yml             # M16
├── requirements.txt
├── pytest.ini                     # M-T (integration marker)
└── README.md
```

> Papkalarni **M1** da bir yo'la yaratib qo'y (bo'sh `__init__.py` bilan). Fayllarni esa faqat o'z milestone'ida yoz — oldindan to'ldirma.

**Hozirgi holat (M0–M1 tugagan):** barcha papkalar, `docs/architecture.md`, `docs/io_contract.md`, `scripts/make_sample_pdfs.py`, `.env.example`, `.gitignore`, `requirements.txt` (versiyalar qotirilgan), `.venv` (Python 3.12.14), `data/raw/` da 4 ta namuna PDF mavjud.

---

# 5. Texnologiyalar

| Kutubxona | Vazifa | Qaysi milestone |
|-----------|--------|-----------------|
| **PyMuPDF** (`import pymupdf` yoki `import fitz`) | PDF matnini olish | M2 |
| **NumPy** | sonli hisoblar, manual implementatsiyalar | M5–M9 |
| **pandas** | jadval ko'rinishidagi tahlil (cluster statistikasi) | M6–M7 |
| **scikit-learn** | TF-IDF, KMeans, silhouette, PCA/SVD | M5–M8 |
| **matplotlib** | grafiklar (elbow, PCA scatter) | M7–M8 |
| **sentence-transformers** | semantik embedding | M9 |
| **Ollama** | lokal LLM | M11 |
| **FastAPI + Uvicorn** | backend | M13 |
| **Pydantic** | sxemalar/validatsiya | M2+ |
| **Streamlit** | frontend | M14 |
| **pytest** (+ pytest-cov) | testlar | M15 |
| **Docker** | deploy | M16 |
| **torch (CPU-only)** | embedding uchun fon (GPU kerak emas) | M9 |
| **uv** | tez venv/pip boshqaruvi (ixtiyoriy, pip o'rniga) | M1 |
| **googletrans** (ixtiyoriy) | M-T test/tajriba tarjimasi | M-T |

`requirements.txt` **M1 da versiyalar bilan qotirilgan** (`.venv` da o'rnatilgan holatdan `pip freeze` asosida).

**Muhim:** `torch` GPU versiyasi juda katta (~2.5 GB+). Shu loyiha uchun **CPU-only** torch yetarli:

```bash
# Muhit yaratish (M1 da bajarilgan)
uv venv --python 3.12 .venv

# torch — ALMASHIDA CPU-only indeksidan
uv pip install --python .venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cpu

# qolgan kutubxonalar
uv pip install --python .venv/bin/python -r requirements.txt
```

> **Python versiyasi:** loyiha `Python 3.12` da ishlaydi (`Python 3.14` da `torch`/`scikit-learn` wheel'lar hali yo'q — M1 da aniqlanadi va `.venv` da `3.12` qo'yiladi).

**O'rnatilgan versiyalar (M1 holati):**

| Kutubxona | Versiya |
|-----------|---------|
| Python | 3.12.14 |
| pymupdf | 1.28.2 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| sentence-transformers | 6.1.0 |
| torch | 2.14.1+cpu |
| fastapi | 0.142.2 |
| streamlit | 1.65.0 |
| pydantic | 2.13.5 |

---

# 6. Loyiha tamoyillari

1. **Abstraksiyadan oldin tushunish.** Har muhim ML komponenti 2.2-bo'limdagi 10 qadamdan o'tadi.
2. **Inkremental qurish.** Har milestone oxirida *ishlaydigan natija* bo'lishi shart.
3. **ML sifatini o'lcha.** Model xatosiz ishlashi ≠ natija ma'noli. Raqam + qo'lda tekshiruv.
4. **Komponentlarni ajrat.** PDF, preprocessing, ML, LLM, API — alohida modullar.
5. **LLM unsupervised model emas.** U faqat tushuntiruvchi qatlam.
6. **Reproducibility.** Tasodifiy joylarda doim `random_state=42`.
7. **Kichikdan boshla.** Avval 5–10 betlik PDF, keyin kattasi.

---

# 7. Milestone'lar

Taxminiy vaqtlar — kuniga 1.5–2 soat ishlaganda. Bu *taxmin*, qat'iy muddat emas; tushunmasdan o'tishdan ko'ra sekin yurgan yaxshi.

| M | Mavzu | Taxminiy vaqt | Natija |
|---|-------|---------------|--------|
| M0 | Planning | 0.5–1 kun | Loyiha spetsifikatsiyasi ✅ |
| M1 | Environment | 0.5–1 kun | Ishlaydigan muhit ✅ |
| **M-T** | **Translation Module** | **2–3 kun** | **Pluggable tarjima (ixtiyoriy)** |
| M2 | PDF extraction | 1–2 kun | Sahifa-darajadagi matn |
| M3 | Preprocessing | 2–3 kun | Tozalangan matn |
| M4 | Chunking | 2–3 kun | Chunk'lar ro'yxati |
| M5 | TF-IDF | 3–4 kun | Sonli vektorlar |
| M6 | K-Means | 4–5 kun | Cluster'lar |
| M7 | Cluster evaluation | 2–3 kun | Tanlangan `k` + isbot |
| M8 | PCA/SVD | 1–2 kun | 2D vizualizatsiya |
| M9 | Embeddings | 3–4 kun | Semantik vektorlar |
| M10 | Topic discovery | 3–4 kun | Interpretatsiya qilingan mavzular |
| M11 | Ollama | 2–3 kun | Lokal LLM aloqa |
| M12 | Full pipeline | 2–3 kun | Bitta funksiya — butun tahlil |
| M13 | FastAPI | 2–3 kun | HTTP API |
| M14 | Streamlit | 1–2 kun | UI |
| M15 | Testing | 3–4 kun | Avtomatik testlar |
| M16 | Docker | 2–3 kun | Konteynerlar |
| M17 | Final evaluation | 2–3 kun | Baholash hisoboti |
| M18 | Production cleanup | 2–3 kun | Toza loyiha |

**Umumiy:** taxminan 6–9 hafta.

> Har milestone shu shablon bo'yicha yozilgan: **Maqsad → Oldindan nima kerak → O'rganish → Qayerda ishlaysan → Qadamlar → Kod skeleti → Qanday tekshirasan → Tipik xatolar → Definition of Done → Commit.**

---

## M0 — Loyiha rejalashtirish

**Maqsad:** aniq nima qurayotganingni hujjatlashtirish.
**Oldindan kerak:** hech narsa.
**Qayerda:** `docs/architecture.md` (yoki shu README).

### Task breakdown

> Har task **2.2-bo'limdagi 8 bosqichli workflow** bo'yicha bajariladi: Understand → Inspect → Plan → Implement → Test → Review → Validate → Report.

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M0.T1** | Scope, Goals, Non-Goals, arxitektura brief | Docs | — | `docs/architecture.md` |
| **M0.T2** | Input/Output contract + baholash mezonlari | Docs | M0.T1 | `docs/io_contract.md` |
| **M0.T3** | Namuna PDF'lar (4 ta, aralash tilli bilan) | Data | M0.T1 | `scripts/make_sample_pdfs.py`, `data/raw/` |
| **M0.T4** | **ADR'lar** — 5 ta arxitektura qarori | Decision | M0.T1 | `docs/architecture/ADR-001..005*.md` |

**Tugallandi:** ✅ M0.T1, ✅ M0.T2, ✅ M0.T3, ✅ M0.T4

### ADR'lar (`system-architect.md` §43 — majburiy)

Arxitektura qarorlari **jamiyat xotirasida** saqlanadi. Quyidagi 5 ta qaror qabul qilindi:

| ADR | Qaror | Nima uchun |
|-----|--------|------------|
| [ADR-001](docs/architecture/ADR-001-kmeans-over-bertopic.md) | TF-IDF + K-Means, BERTopic emas | Pipeline'ning har bosqichini o'zlashtirish — asosiy maqsad |
| [ADR-002](docs/architecture/ADR-002-llm-local-ollama-pluggable.md) | Ollama default, pluggable LLM | Offline + bepul + kelajakka yo'l |
| [ADR-003](docs/architecture/ADR-003-translation-pluggable-page-level.md) | Pluggable translation, default ochiq, faqat page/document | Chunk level kontekstni buzadi |
| [ADR-004](docs/architecture/ADR-004-cpu-only-torch-python-312.md) | Python 3.12 + CPU-only torch | 3.14 da wheel yo'q; GPU kerak emas |
| [ADR-005](docs/architecture/ADR-005-single-container-monolith.md) | FastAPI monolith + Streamlit | Monolith-first (`system-architect` §7) |

**ADR shabloni** (majburiy, §43): `Context → Problem → Options Considered → Decision → Trade-offs → Consequences → Status`

**Yangi ADR qachon yoziladi:** texnologiya o'zgarishi, service chegarasi, autentifikiya, breaking API o'zgarishi, katta infratuzilma, caching, tarjima plug-in kontrakti.
### Qadamlar
> ✅ **Bajarildi** — natija `docs/architecture.md` (20 bo'lim) va `docs/io_contract.md`.

- [x] **Scope**: "nima bor / nima yo'q" ni 5–10 qatorda yoz (11-bo'limdagi non-goals'ni ko'chir).
- [x] **MVP** ni yoz:
  ```text
  PDF yukla → matn ol → tozala → chunk'la → vektorla → cluster'la
            → mavzularni top → Ollama → xulosa
  ```
- [x] **Input/Output** — to'liq contract `docs/io_contract.md` da:
  - Input: bitta PDF fayl (matnli, ≤ ~100 bet). Validatsiya: magic bytes `%PDF`, hajm, `needs_pass`, matn borligi.
  - Output: `main_topic`, `topics[]`, `clusters[]`, `summary`, `key_points[]`, `important_sections[]` + `stats`, `warnings`.
- [x] **Baholash mezonlari** (8-bo'lim + `docs/io_contract.md` §3): silhouette, inertia, cluster o'lchamlari, stability, qo'lda tekshiruv (rubric K1–K3).
- [x] **Cheklovlar**: skanerlangan PDF (OCR yo'q), faqat matnli PDF, ingliz asosiy til + **multilingual pipeline** (M-T).
- [x] **Namuna PDF'lar** `data/raw/` ga qo'yildi (`scripts/make_sample_pdfs.py` orqali yaratilgan, 4 ta):
  - 1 ta qisqa (5–10 bet) — tajriba uchun
  - 1 ta o'rta (20–40 bet) — asosiy test
  - 1 ta ko'p mavzuli (kitob boblari/hisobot) — clustering sifatini ko'rish uchun
  - 1 ta aralash-tilli (o'zbek + ingliz + rus) — translation sinovi uchun

  | Fayl | Betlar | Belgi | Maqsad |
  |------|--------|-------|--------|
  | `01_short_ml_basics.pdf` | 3 | 2 068 | Qisqa — tajriba |
  | `02_medium_distributed_systems.pdf` | 8 | 4 928 | O'rta — asosiy test |
  | `03_multi_topic_report.pdf` | 12 | 6 455 | Ko'p mavzuli — clustering sifati |
  | `04_mixed_language.pdf` | 6 | 3 042 | Aralash tilli — M-T sinovi (561 Kirill, 562 o'zbekcha harfi) |

  > **Generator shrifti:** `fontfile` bilan DejaVuSans. `helv` (Helvetica) **faqat lotin**
  > harflarni qo'llaydi — ruscha matn `??????` bo'lib chiqardi va tarjima uni tarjima
  > qila olmasdi. M-T da aniqlangan va tuzatilgan nuqson.

### Definition of Done
- Scope, arxitektura, MVP, tartib hujjatlashtirilgan.
- `data/raw/` da kamida 2 ta namuna PDF bor.

**Commit:** `M0: project spec and sample PDFs`

---

## M1 — Muhit sozlash

**Maqsad:** takrorlanadigan (reproducible) Python muhiti.
**Oldindan kerak:** Python 3.10+, Git, Ollama o'rnatilgan.
**Qayerda:** loyiha ildizi.

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M1.T1** | venv (Python 3.12) + `requirements.txt` o'rnatish | Setup | — | `requirements.txt`, `.venv/` |
| **M1.T2** | Papka tuzilmasini yaratish (`app/translation/` bilan) | Setup | M1.T1 | `app/**/__init__.py` |
| **M1.T3** | `.gitignore` + `.env.example` (translation + LLM provider) | Config | M1.T2 | `.gitignore`, `.env.example` |
| **M1.T4** | Muhit + Ollama tekshiruvi | Verify | M1.T1–T3 | — (terminal) |

**Tugallandi:** ✅ M1.T1, ✅ M1.T2, ✅ M1.T3, ✅ M1.T4

> **M1 da aniqlangan muammo:** sistem Python **3.14.4** edi — `torch`/`scikit-learn` uchun wheel yo'q. Yechim: `uv venv --python 3.12 .venv` (3.12.14) + **CPU-only torch** (`--index-url .../whl/cpu`, 2.5 GB → ~200 MB).
> **M1 da aniqlangan fakt:** Ollama'da `llama3.2:3b` **yo'q**. Mavjud: `qwen3:4b`, `qwen3:8b`, `nomic-embed-text`, `mxbai-embed-large`. `.env.example` da `OLLAMA_MODEL=qwen3:4b` qo'yildi (kichik = tez; `qwen3:8b` sifatliroq, keyinchalik almashtiriladi).

### Qadamlar
> ✅ **Bajarildi** — barcha quyidagilar bajarilgan va tekshirilgan.

- [x] Git repo (allaqachon mavjud):
  ```bash
  git init        # ✅ bajarilgan
  ```
- [x] Virtual environment (**Python 3.12**, CPU-only torch):
  ```bash
  uv venv --python 3.12 .venv                              # ✅ 3.12.14
  uv pip install --python .venv/bin/python torch \
      --index-url https://download.pytorch.org/whl/cpu     # ✅ CPU-only
  ```
- [x] `requirements.txt` yoz (versiya bilan qotirilgan) va o'rnat:
  ```bash
  uv pip install --python .venv/bin/python -r requirements.txt   # ✅
  ```
- [x] 4-bo'limdagi papka tuzilmasini yarat (`app/translation/` bilan):
  ```bash
  mkdir -p app/{api,core,ml,nlp,llm,services,schemas,translation} \
           tests/fixtures notebooks data/{raw,processed} docs/figures frontend scripts
  # ✅ barcha papkalar + bo'sh __init__.py lar
  ```
- [x] `.gitignore`:
  ```gitignore
  .venv/
  __pycache__/
  *.pyc
  .env
  .ipynb_checkpoints/
  data/processed/
  data/raw/*.pdf        # katta PDF'larni commit qilma (kichik fixture'lar tests/fixtures da)
  .pytest_cache/
  ```
- [x] `.env.example`:
  ```env
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_MODEL=llama3.2:3b
  EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
  MAX_UPLOAD_MB=25
  ```
- [x] **Python'ni tekshir:** `python --version` → **3.14.4** (⚠️ mos emas, quyida `uv` bilan 3.12 yaratiladi)
- [x] **Kutubxonalarni tekshir:**
  ```bash
  python -c "import pymupdf, sklearn, numpy, pandas, fastapi, streamlit; print('OK')"
  ```
- [x] **Ollama'ni tekshir:**
  ```bash
  ollama --version
  ollama pull llama3.2:3b          # yoki qurilmangga mos boshqa model
  ollama run llama3.2:3b "Say hi"
  curl http://localhost:11434/api/tags
  ```
- [x] Birinchi README va commit.

### Tipik xatolar
- `pip` global Python'ga o'rnatadi → `which python` `.venv` ichini ko'rsatayotganini tekshir.
- Ollama ishlamayapti → `ollama serve` ni alohida terminalda yoq.

### Definition of Done
- [x] `.venv` (Python 3.12.14) + CPU-only torch o'rnatilgan
- [x] Barcha kutubxonalar import qilinadi (`pymupdf`, `sklearn`, `numpy`, `pandas`, `fastapi`, `streamlit`, `sentence_transformers`, `torch`)
- [x] Ollama ishlayapti: `/api/tags` → `qwen3:4b`, `/api/generate` → javob berdi
- [x] `.gitignore` to'g'ri ishlaydi (`.venv/`, `data/raw/*.pdf`, `.env`, `data/processed/` — barchasi ignored)
- [x] `app/**` va `docs/` — tracked

**Commit:** `M1: environment, structure, requirements`

---

---

## M-T — Translation Module (pluggable, ixtiyoriy)

**Maqsad:** turli tilli PDF'larni qo'llab-quvvatlash uchun **pluggable** tarjima moduli. Tarjima **majburiy emas** — config orqali boshqariladi, default **OCHIQ** (local-first).

**Oldindan kerak:** M1.T3 (config maydonlari).
**Qayerda:** `app/translation/`, `tests/test_translation.py`, `notebooks/11_translation_experiments.ipynb`.

> **Nima uchun alohida milestone?** Chunki bu — **butun pipeline'ga kirishadigan tashqi servis**. M5 (stop_words), M9 (embedding tanlovi), M12 (pipeline), M13 (xato xaritasi), M17 (baholash) — hammasi shu modulga bog'liq. Uni alohida qurish, keyin ulash — **arxitektura toza qoladi**.

### Qanday ishlaydi

```text
Option A (default):  PDF → extract → [preprocess] → chunk → multilingual embedding
Option B:             PDF → extract → TRANSLATE (page/document) → [preprocess] → chunk → monolingual embedding
```

### Qat'iy qoidalar

1. **Hech qachon chunk darajasida tarjima qilinmaydi** — faqat `document` yoki `page`.
2. Tarjima **pre-chunk** (sahifa darajasida).
3. Default — `NoopTranslator` (tashqi so'rov **umuman yuborilmaydi**).
4. **Interface-first** — yangi provider `base.py` ni tegilmasdan qo'shiladi.
5. **Graceful** — provider ishlamasa `TranslationUnavailableError` → HTTP **503**.

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M-T.T1** | `exceptions.py` + `base.py` (Protocol/ABC) | Core | M1.T3 | `app/translation/exceptions.py`, `app/translation/base.py` |
| **M-T.T2** | `NoopTranslator` (default, passthrough) | Core | M-T.T1 | `app/translation/noop.py` |
| **M-T.T3** | `GoogleTranslateTranslator` (httpx, retry, timeout) | Integration | M-T.T2 | `app/translation/google_translate.py` |
| **M-T.T4** | `factory.py` + `__init__.py` public API | Core | M-T.T3 | `app/translation/factory.py`, `app/translation/__init__.py` |
| **M-T.T5** | `config.py` ga translation maydonlari | Config | M-T.T4 | `app/core/config.py` |
| **M-T.T6** | Unit testlar (identity, factory, mock HTTP, error) | Tests | M-T.T5 | `tests/test_translation.py` |
| **M-T.T7** | Notebook: original vs translated solishtirish | Experiments | M-T.T6 | `notebooks/11_translation_experiments.ipynb` |

**Tugallandi:** ✅ M-T.T1 → ✅ M-T.T7

**Yaratilgan fayllar:**

```text
app/translation/
├── exceptions.py       # TranslationError + 3 ta kichik xato
├── base.py             # Translator Protocol, BatchMode, validate_batch_mode
├── noop.py             # NoopTranslator (default)
├── google_translate.py # httpx + cheklangan retry + timeout
├── factory.py          # get_translator()
└── __init__.py         # public API

app/core/config.py      # Settings (pydantic-settings)
tests/test_translation.py   # 53 ta test
pytest.ini              # 'integration' marker ro'yxatga olingan
```

### Kod skeleti — `app/translation/base.py`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Translator(Protocol):
    name: str
    enabled: bool

    def translate_pages(self, texts: list[str], target: str) -> list[str]:
        # Sahifa (yoki document) darajasida tarjima qiladi
        ...

    def healthcheck(self) -> bool:
        # Provider mavjudligini tekshiradi
        ...
```

### Kod skeleti — `app/translation/factory.py`

```python
def get_translator(settings=None) -> Translator:
    # Config asosida translator instance qaytaradi
    if not settings.translation_enabled:
        return NoopTranslator()
    if settings.translation_provider == "google_translate":
        return GoogleTranslateTranslator(...)
    raise TranslationError(f"Noma'lum provider: {settings.translation_provider}")
```

### Qanday tekshirilgan

Barchasi haqiqiy bajarildi (2026-10-04):

| Tekshiruv | Natija |
|----------|--------|
| `pytest -m "not integration"` | ✅ **53 passed**, 0 failed |
| Coverage `app/translation` | **94%** (9 qator qoldi — `_close`, `__repr__`, oxirgi raise) |
| `pytest -m integration` (haqiqiy tarmoq) | ✅ **1 passed** |
| Notebook 11 `nbconvert --execute` | ✅ **10/10 kod hujjati**, xatosiz |
| `NoopTranslator.translate_pages(["salom"])` | ✅ `["salom"]` — identik |
| Factory `TRANSLATION_ENABLED=false` | ✅ `NoopTranslator` |
| Chunk level rad etish | ✅ `TranslationConfigError` (config + konstruktor + `validate_batch_mode`) |
| 429 → retry → muvaffaqiyat | ✅ 2 urinish |
| 503 → 3 urinish → `TranslationUnavailableError` | ✅ |
| 400 → retry **yo'q**, `TranslationResponseError` | ✅ |
| Buzilgan javob (10 variant) | ✅ `TranslationResponseError` |
| Bo'sh sahifa → so'rov yuborilmaydi | ✅ 0 so'rov |
| Sahifa tartibi saqlanadi | ✅ |
| Haqiqiy tarjima (6 sahifa, o'zbekcha+ruscha) | ✅ 0.1 s/sahifa |
| Texnik atomalar saqlanadi | ✅ 7/8 (`embedding` PDF'da umuman yo'q) |

**O'lchovlar (notebook 11 dan):**

| Ko'rsatkich | Natija |
|-------------|--------|
| Sahifalar | 6 (3042 belgi) |
| Kirill harflari | 561 (37 xil) — `?` **0** |
| Tarjima tezligi | 0.1 s/sahifa |
| Matn kengayishi | 0.99x (deyarli o'zgarmas) |
| Texnik atoma saqlanishi | 7/8 |

### Tipik xatolar

- **Chunk level tarjima** → kontekst yo'qoladi, terminologiya buziladi, so'rovlar ko'payadi. **Taqiqlangan.**
- **Tekislik (flat) matn berish** → sahifa chegarasi yo'qoladi, keyingi bosqichda qayta tiklash mumkin emas.
- **Rate limit** → retry + exponential backoff kerak.
- **Default'ni `true` qoldirish** → maxfiylik buziladi, xarajat oshadi, offline ishlamaydi.

### Definition of Done
- [x] `app/translation/` moduli interface-first (`base.py` — Protocol)
- [x] Default `NoopTranslator` — tashqi so'rov **yuborilmaydi** (test bilan tekshirilgan)
- [x] Faqat `document`/`page` level — chunk level **kod darajasida** taqiqlangan
- [x] Config bilan boshqariladi (`app/core/config.py`, `.env.example`)
- [x] Timeout majburiy, retry **cheklangan** (infinite loop yo'q)
- [x] Testlangan: 53 unit + 1 integration, 94% coverage
- [x] Notebook: original vs tarjima solishtirildi va o'lchandi
- [x] Xato → `TranslationUnavailableError` (M13 da 503 ga aylanadi)
- [ ] **M12 uni pipeline'ga ulaydi** — keyingi milestone

**M-T da aniqlangan muammo (tuzatildi):** `scripts/make_sample_pdfs.py` `fontname="helv"`
(Helvetica) ishlatardi — u **faqat lotin** harflarni qo'llaydi. Ruscha matn `??????` bo'lib
yozilardi (570 ta `?`, 0 ta Kirill) va tarjima moduli uni tarjima **qila olmasdi**.
Yechim: `fontfile` bilan DejaVuSans (Kirill + o'zbekcha `Ўҳ` to'liq qo'llaydi).

**Commit:** `M-T: add pluggable translation module (noop/google, page-level)`

---

## M2 — PDF matnini ajratib olish

**Maqsad:** PDF → sahifa-darajadagi matn.
**Oldindan kerak:** M1.
**Qayerda:** tajriba `notebooks/01_pdf_extraction.ipynb` → yakuniy `app/services/pdf_extractor.py`, xatolar `app/core/exceptions.py`, sxema `app/schemas/document.py`.

### O'rganish
- PDF "matn qatlami" nima? Skanerlangan PDF'da nega matn yo'q (u rasm)?
- PyMuPDF'da `page.get_text("text")` va `page.get_text("blocks")` farqi.

### Ma'lumot ko'rinishi
```python
{"page": 1, "text": "..."}
```

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M2.T1** | Notebook: PyMuPDF bilan tanishish | Experiments | M1 | `notebooks/01_pdf_extraction.ipynb` |
| **M2.T2** | `PageText` sxemasi + domain exceptions | Core | M2.T1 | `app/schemas/document.py`, `app/core/exceptions.py` |
| **M2.T3** | `extract_pages()` — sahifa matnini olish | Service | M2.T2 | `app/services/pdf_extractor.py` |
| **M2.T4** | Unit testlar (valid/empty/scanned/corrupted) | Tests | M2.T3 | `tests/test_pdf_extractor.py` |

**Tartib:** T1 → T2 → T3 → T4. T3 da `Magic bytes` (%PDF) va `needs_pass` tekshiruvi **majburiy** (I/O contract §1.2).
### Qadamlar
- [ ] Notebook'da PDF och: `doc = pymupdf.open(path)`; `len(doc)`; `doc.metadata`.
- [ ] Sahifalarni aylanib chiq, `page.get_text()` ol.
- [ ] Sahifa raqamlarini saqla (1 dan boshlab — odamga qulay).
- [ ] Bo'sh sahifalarni aniqla (`text.strip() == ""`) — o'tkazib yubor, lekin hisobla.
- [ ] Xatolarni ushla: fayl topilmadi, buzilgan PDF, parol bilan himoyalangan.
- [ ] **Skanerlangan PDF aniqlash:** agar barcha sahifalardagi jami matn juda kam (masalan < 100 belgi) → `ScannedPdfError` ko'tar.
- [ ] Notebook'da 3 ta namuna PDF'da sina; sahifa va belgi sonlarini chiqar.

### Kod skeleti — `app/schemas/document.py`
```python
from pydantic import BaseModel

class PageText(BaseModel):
    page: int
    text: str
```

### Kod skeleti — `app/core/exceptions.py`
```python
class PdfAnalyzerError(Exception): ...
class InvalidPdfError(PdfAnalyzerError): ...
class ScannedPdfError(PdfAnalyzerError): ...
class NotEnoughContentError(PdfAnalyzerError): ...
```

### Kod skeleti — `app/services/pdf_extractor.py`

> ⚠️ Bu skelet **qisqartirilgan**. To'liq implementatsiya qo'shadi:
> magic bytes (I1), bo'sh fayl (I2), `needs_pass` (I5), sahifa soni (I7),
> hujjatni `finally` da yopish.

```python
def extract_pages(
    source: str | Path | bytes, *, min_total_chars: int | None = None
) -> list[PageText]:
    if min_total_chars is None:
        min_total_chars = get_settings().min_total_chars   # .env dan, kodga yozilmaydi
    _validate_head(source)      # I1, I2 — magic bytes va bo'sh fayl
    doc = _open(source)         # I6 — pymupdf xatolari -> InvalidPdfError
    try:
        _validate_document(doc)  # I5, I7 — parol va sahifa soni
        return _read_pages(doc, min_total_chars)   # I4 — matn chegarasi
    finally:
        doc.close()             # descriptor oqib ketmasin
```

**Tekshiruv zanjiri** (`docs/io_contract.md` §1.2):

| # | Qoida | Xato |
|---|-------|------|
| I1 | magic bytes `%PDF` | `InvalidPdfError` |
| I2 | fayl bo'sh emas | `InvalidPdfError` |
| I4 | jami belgi >= `min_total_chars` | `ScannedPdfError` |
| I5 | `needs_pass == False` | `InvalidPdfError` |
| I6 | `pymupdf.open()` muvaffaqiyatli | `InvalidPdfError` |
| I7 | kamida 1 sahifa | `InvalidPdfError` |

### Qanday tekshirilgan

Barchasi haqiqiy bajarildi (2026-10-04):

| Tekshiruv | Natija |
|-----------|--------|
| `pytest tests/test_pdf_extractor.py` | ✅ **45 passed** |
| Coverage `app/services/pdf_extractor.py` | **100%** |
| Coverage `app/schemas/document.py` | **100%** |
| Coverage `app/core/exceptions.py` | **100%** |
| To'liq suite (`-m "not integration"`) | ✅ **98 passed** (53 M-T + 45 M2) |
| Notebook 01 `nbconvert --execute` | ✅ **13/13 kod hujjati**, 0 xato |
| 4 ta namuna PDF (path) | ✅ 3/8/12/6 sahifa — 2065/4920/6443/3036 belgi |
| `bytes` kirish == `path` kirish | ✅ bit-bit bir xil |
| Sahifa raqamlari 1-based, tartibi saqlanadi | ✅ |
| Bo'sh sahifalar **o'chirilmaydi** | ✅ `[True, False, True]` |
| 10 ta xato holati | ✅ to'g'ri xato turi |
| Xatoda hujjat yopiladi (`doc.is_closed`) | ✅ |

**Xato holatlarining xaritasi:**

| Holat | Natija |
|-------|--------|
| fayl yo'q | `InvalidPdfError` |
| manba papka | `InvalidPdfError` |
| bo'sh fayl (0 bayt) | `InvalidPdfError` |
| `.pdf` nomli PNG | `InvalidPdfError` (magic bytes) |
| oddiy matn, `.pdf` kengaytmasi | `InvalidPdfError` (magic bytes) |
| buzilgan PDF (header butun) | `InvalidPdfError` (`pymupdf`) |
| parol bilan himoyalangan | `InvalidPdfError` (`needs_pass`) |
| sahifasi yo'q PDF | `InvalidPdfError` (I7) |
| ruxsat rad etilgan fayl | `InvalidPdfError` |
| skanerlangan (matn qatlami yo'q) | `ScannedPdfError` |
| o'qilgan, lekin 100 belgidan kam | `ScannedPdfError` |

### M2 da aniqlangan cheklovlar

`docs/limitations.md` yaratildi (M17 da kengaytiriladi). Faqat **o'lchov
bilan tasdiqlangan** topilmalar:

| ID | Cheklov | Qaror |
|----|---------|-------|
| **L-01** | `sort=True` ikki ustunli matnni buzadi | `sort=True` ishlatilmaydi + regressiya testi |
| **L-02** | Skanerlangan PDF | `ScannedPdfError` (OCR — non-goal) |
| **L-03** | Parol bilan PDF | `InvalidPdfError` (`needs_pass`) |
| **L-04** | `MIN_TOTAL_CHARS` chegarasi ikki tomonlama xato qiladi | 100, `.env` dan o'qiladi |
| **L-05** | Testlar `data/raw/` ga bog'liq emas | PDF'lar test ichida yaratiladi |
| **L-06** | `insert_text` test fixture matnini kesadi | `insert_textbox` ishlatiladi |
| **L-07** | `ENGLISH_STOP_WORDS` `system` ni o'ldiradi | `PROTECTED_TERMS` (75 ta) |
| **L-08** | Namuna PDF'lar dekorativ chiziqdan **majburiy** tozalanadi | `remove_decorative_rules` qo'shildi |
| **L-09** | Stop-word filtrining o'zi kichik korpusda ma'nosiz | korpus o'lchovi bilan tekshirildi |
| **L-10** | Stemming o'lchangan, lekin Porter emas | B (stemming'siz) tanlandi |
| **L-11** | Stop-word faqat inglizcha | `va`, Kirill saqlanadi |
| **L-12** | `fix_hyphenation` so'z **yaratib** yubaradi | `build_vocabulary()` bilan tasdiqlash |
| **L-13** | Ligatura lug'atni ifloslantiradi | `build_vocabulary` normalizatsiya qiladi |
| **L-14** | Chekka izlash real PDF header'ini topmaydi | `edge_lines=None` **ishlatilmaydi** (509 qator yo'q qilinadi) |
| **L-15** | Boshqa kitob yuklangan (ID 25438 ≠ The Jungle) | `verify_title()` — sarlavha matndan tekshiriladi |

> **Eski maslahat bekor qilindi:** "ikki ustunli PDF'da matn tartibi
> buziladi → `get_text("blocks", sort=True)` ni sinab ko'r" — o'lchovda
> noto'g'ri chiqdi (`docs/limitations.md` L-01).

### Tipik xatolar
- ⚠️ **`get_text("sort=True")` ishlatmang.** U matnni vertikal koordinat
  bo'yicha saralaydi va **ikki ustunli** hujjatda chap/o'ng ustunni qatorlab
  aralashib ketiradi — ya'ni ishlaydigan yagona variantni buzadi. Standart
  `get_text("text")` (content stream tartibi) ishlatiladi. Ikkala holat ham
  o'lchab tasdiqlangan → `docs/limitations.md` **L-01**.
- Ligatura/maxsus belgilar (`ﬁ`, `ﬂ`) → M3 da normallashtiriladi.

### Definition of Done
- [x] `extract_pages()` sahifa-darajadagi matn qaytaradi (1-based)
- [x] Bo'sh sahifalar **o'chirilmaydi** — statistika uchun saqlanadi
- [x] Bo'sh/buzilgan/skanerlangan/parol bilan PDF uchun **aniq** xato
- [x] Magic bytes `%PDF` **har doim** tekshiriladi (I1)
- [x] `needs_pass` alohida tekshiriladi (I5) — skanerlangan bilan aralashmasin
- [x] `min_total_chars` **kodga yozilmagan** — `.env` dan o'qiladi (qa §18)
- [x] `path`, `Path` va `bytes` kirishlari bir xil semantika
- [x] Hujjat xatoda ham yopiladi (`finally`)
- [x] Testlangan: 45 test, `pdf_extractor.py` **100%** coverage
- [x] Cheklovlar hujjatlashtirilgan (`docs/limitations.md`)
- [ ] **M3 uni preprocessing'ga ulaydi** — keyingi milestone

**Commit:** `M2: pdf extraction with error handling`

---

## M3 — Matnni oldindan qayta ishlash (Preprocessing)

**Maqsad:** xom matnni ML uchun izchil, toza ko'rinishga keltirish.
**Qayerda:** `notebooks/02_preprocessing.ipynb` → `app/nlp/preprocessing.py`.

### O'rganish
Tokenizatsiya, lowercasing, stop words, tinish belgilari, bo'shliqlarni normallashtirish, stemming, lemmatization, shovqinni olib tashlash.

> ⚠️ **Texnik atamalarni ko'r-ko'rona o'chirma:** `AI`, `API`, `SQL`, `CNN`, `RAG` muhim ma'lumot tashiydi. Stop-word ro'yxatiga ularni qo'shma; lowercasing'dan keyin ham yo'qolmasligini tekshir.

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M3.T1** | Notebook: artefaktlarni ko'rish + 3 strategiya | Experiments | M2 | `notebooks/02_preprocessing.ipynb` |
| **M3.T2** | `preprocessing.py` — 6 ta kichik funksiya + `clean_text` | NLP | M3.T1 | `app/nlp/preprocessing.py` |
| **M3.T3** | `display_text` / `ml_text` ajratish | NLP | M3.T2 | `app/nlp/preprocessing.py` |
| **M3.T4** | Unit testlar (idempotentlik, texnik atamalar) | Tests | M3.T3 | `tests/test_preprocessing.py` |

**Muhim:** M3.T3 — M9 ning oldindan sharti. Embedding uchun `display_text`, TF-IDF uchun `ml_text` ishlatiladi.
### Qadamlar
- [x] **Extraction artefaktlarini top** (notebook'da 5–10 sahifani ko'z bilan ko'r):
  - so'z o'rtasidagi defis bilan uzilish: `informa-\ntion` — **namuna PDF'larida yo'q**
  - takrorlanuvchi header/footer (har sahifada bir xil qator) — **✅ 4/4 faylda**
  - sahifa raqamlari (`12`, `Page 12 of 40`) — **✅ 4/4 faylda**
  - ortiqcha bo'sh qatorlar, `\\xa0`, ligaturalar — **namuna PDF'larida yo'q**
  - havolalar/URL, `[12]` kabi iqtiboslar — **namuna PDF'larida yo'q**
- [x] **Funksiyalar yoz** (har biri alohida, kichik, testlanadigan):
  - `normalize_whitespace(text)`
  - `fix_hyphenation(text)` — `-\n` ni birlashtirish
  - `remove_repeated_headers_footers(pages)` — kamida 50% sahifada uchraydigan qisqa qatorlarni o'chir
  - `remove_page_numbers(text)`
  - `normalize_unicode(text)` — `unicodedata.normalize("NFKC", ...)` ligaturalarni tuzatadi
  - `clean_text(text)` — yuqoridagilarning ketma-ketligi
- [x] **Muhim:** ikki xil matn kerak bo'lishi mumkin:
  - `display_text` — yengil tozalangan (chunk'lar foydalanuvchiga/LLM'ga ko'rsatiladi)
  - `ml_text` — ancha tozalangan (vektorlash uchun)

  Ikkalasini ham saqla, aks holda LLM'ga lowercased/stop-word'siz matn yuborasan.
- [x] **Strategiyalarni solishtir** (kamida 3 xil) va natijani jadvalga yoz:

  O'lchov korpusi: 29 sahifa / 13 869 belgi (4 ta namuna PDF, artefaktlar olib tashlangandan keyin).

  | Strategiya | Stop words | Lowercase | Stemming | Token | Lug'at | Top-10 so'z ma'noli-mi? |
  |------------|-----------|-----------|----------|-------|--------|---------------------------|
  | A: minimal | yo'q | ha | yo'q | 1 817 | 926 | ❌ `and, the, a, of, to, as` — F7 muammosi |
  | **B: standart** | **ha** | **ha** | yo'q | **1 393** | **854** | ✅ `chapter, data, nodes, cluster, replication` |
  | C: agressiv | ha | ha | ha* | 1 393 | 775 | ✅ lekin atoma buziladi: `kmeans→kmean`, `embedding→embedd` |

  **Tanlov: B.** C lug'atni yana 9.3% qisqartiradi, lekin texnik atomani
  buzadi — bu qiymat bermaydi, chunki muammo token ko'pligi emas,
  **ma'nosiz token**.

  *`crude_stem` — ataylab yozilgan sodda kesuvchi, Porter **emas**; muhitda
  stemmer yo'q. Cheklov: `docs/limitations.md` L-10.*

- [x] 8–10 ta unit-test misoli yoz (keyinroq `tests/test_preprocessing.py` ga o'tadi).

### Topshirilgan natija (M3)

| Natija | Holat |
|--------|-------|
| `normalize_unicode`, `fix_hyphenation`, `remove_page_numbers`, `remove_decorative_rules`, `normalize_whitespace` | ✅ |
| `remove_repeated_headers_footers(pages)` — chekka + takrorlanish + uzunlik | ✅ |
| `display_text` / `ml_text` ajratilgan | ✅ |
| 92 ta test, `app/nlp` coverage **100%** | ✅ |
| `tests/test_real_corpus.py` — 23 ta test (sarlavha/litsenziya tekshiruvi) | ✅ |
| Barcha funksiyalar idempotent (notebook + test) | ✅ |
| Texnik atoma saqlanadi (`AI, API, SQL, CNN, RAG`) | ✅ |

**Rejalashtirilmagan qo'shimcha funksiya:** `remove_decorative_rules`.
Reja 6 ta funksiya ko'rsatgan, lekin o'lchovda namuna PDF'larining
**4/4 ida** `=====` chiziqlari topildi (3–12 marta). Bu chiziq TF-IDF'ga
`=====` token sifatida kirib, top terms'ni ifloslaydi. Sabab va dalil:
`docs/limitations.md` L-08.

**O'lchangan artefaktlar — sintetik korpus** (4 ta namuna PDF, jami 16 464 belgi):

| Artefakt | Namuna PDF'larda | Tozalash |
|----------|-------------------|----------|
| Takrorlanuvchi header | ✅ 4/4 | `remove_repeated_headers_footers` |
| `Page N of M` | ✅ 4/4 | `remove_page_numbers` |
| `=` dekorativ chiziq | ✅ 4/4 | `remove_decorative_rules` |
| Defis bilan uzilgan so'z | ❌ yo'q | `fix_hyphenation` — **faqat sintetik test** |
| Ligatura, `\xa0` | ❌ yo'q | `normalize_unicode` — **faqat sintetik test** |

Belgining 12–17% i olib tashlandi. Tozalashdan keyin korpus
13 869 belgi.

---

### Real korpus bilan tekshiruv (namuna PDF yetarli emasdi)

Yuqoridagi jadval **4 ta sintetik PDF** ga asoslangan edi. Ular
hyphenation va ligatura **yo'q** edi — ya'ni eng qiyin ikki funksiya
faqat qo'lda yozilgan testlar bilan tekshirilgan edi.

Yuklangan: `scripts/fetch_real_documents.py` — **10 ta Project Gutenberg
kitobi** (public domain, turli janr: ayol, fantastika, detektiv, tabiiy
ilmiy, falsafiy, ijtimoiy) + **5 ta ochiq maqola** (3 × arXiv, 2 × PLOS
ONE). Jami **5 501 150 belgi**, 2 527 + 69 sahifa. Manba, litsenziya
va SHA-256 — `data/manifest.json`.

> **Ikkita o'lchov xatosi bo'ldi.** (1) 60 sahifalik namuna bilan
> o'lchab, ligaturani 207, hyphenation nuqtasini 470 deb yozgan edim —
> to'liq korpusda bu **11 594** va **750**. (2) ID 25438 ni "The
> Jungle" deb yozgan edim, lekin u aslida **"The Airlords of Han"**
> edi (to'g'ri ID — **140**). Namuna asosida yoki xotira asosida
> xulosa yozish — "real ma'lumot bilan tekshirdim" degan narsa emas.

**Natijada to'rtta haqiqiy xato topildi** (barchasi tuzatildi):

| # | Xato | Nima bo'lardi | Tuzatish |
|---|------|---------------|----------|
| 1 | `(\\w)-` faqat **bitta belgi** ushlaydi | `transduc-\ntion` da `c-\nt` match bo'lardi. Natija tasodufen to'g'ri chiqar edi | `(\\w+)-` |
| 2 | Har doim barlashtirish | 750 nuqtadan **446** ta haqiqiy kompaniya buzilardi (`bookshelves`, `twentyfour`) | `build_vocabulary()` — faqat korpusda tasdiqlangan shakl |
| 3 | `build_vocabulary` normalizatsiya qilmasdi | lug'atda `suﬃcient`, solishtiriladigani `sufficient` — **hech qachon** mos kelmasdi | `normalize_unicode` ichida |
| 4 | **Boshqa kitob** yuklangan (ID 25438) | "The Jungle" o'rniga "The Airlords of Han" yuklangan. PDF ochiladi, sahifa va belgi bor — xato **ko'rinmaydi** | `verify_title()` — sarlavha va muallif matn ichidan tekshiriladi |

> Xato 1 faqat xato 2 kiritilganda ko'rindi: yangi kod lug'at bilan
> ishlagan zahoti `re.finditer` ga qarab tushunildi. Yakuniy satrni
> tekshiradigan testlar bu xatoni ko'ra olmasdi.
>
> Xato 4 boshqacha: u **koding emas, ma'lumotning** xatosi — va hech
> qanday birlik test uni ushlamagan, chunki testlar tarmoqqa
> ulanmaydi (L-05). Yechim ma'lumot chegarasida: `verify_title()`
> topilmasa `FetchError` beradi va hujjat manifestga kirmaydi.

**O'lchangan natija** (`clean_pages`, real korpus):

| Ko'rsatkich | Oldin | Keyin |
|-------------|-------|-------|
| Ligatura (butun korpus) | 11 594 | **0** |
| Ligatura (`dracula.pdf`) | 1 691 | **0** |
| Sahifa raqami qolgani | — | **0** |
| `plos_middle_ear_effusion.pdf` | — | **−1.6%** (header/footer) |
| `arxiv_bradley_terry.pdf` | — | **−1.3%** |
| Adabiyot belgisi | — | **+0.1%** (ligatura `ﬀ`→`ff` kengayadi) |
| Lug'at: `dracula`, `mina`, `watson`, `holmes`, `thoreau` | — | ✅ saqlangan |
| Lug'at: `attention`, `transformer`, `retina`, `pca` | — | ✅ saqlangan |

**Hyphenation xarakteri** — to'liq korpus, 750 nuqta:

| Guruh | Sahifa | Nuqta | Bitta so'z sindi | Haqiqiy kompaniya |
|-------|--------|-------|------------------|-------------------|
| Adabiyot | 2 527 | 404 | 60 (15%) | **344 (85%)** — `book-shelves`, `cherry-tart`, `twenty-four`, `star-ﬁsh` |
| arXiv | 39 | 206 | 136 (66%) | 70 (34%) — `sequence-aligned`, `position-wise`, `source-target` |
| PLOS | 30 | 140 | 108 (77%) | 32 (23%) — `A-RC`, `Nose-Throat` |

Bitta so'z sindi (`transduc-` + `tion`) va ikki so'zli kompaniya
(`book-` + `shelves`) matnda **bir xil** ko'rinadi. Konservativ qaror:
**nomutanosib xatolardan yomonini yo'q qilish** — keraksiz
birlashtirish `bookshelves` kabi **so'z yo'q** token yaratadi,
birlashtirmaslik esa `book-shelves` (to'g'ri token) saqlanadi.

**Cheklov:** so'z korpusda hech qayerda butun ko'rinmasa,
birlashtirilmaydi — boshqa ishonchli ro'yxat talab qiladi
(`nltk` yo'q, yangi bog'liqlik qo'shilmadi).

**Arxiv/PLOS header muammosi hal qilinmadi** — `edge_lines=None`
rejimi header'ni topadi (14/14 → 1/14), lekin `arxiv_bradley_terry.pdf`
da **−4.0%**, ya'ni **509 qator** ilmiy mazmun (formula belgilari: `X`,
`i`, `1`) yo'q qiladi. Default `edge_lines=3` qoldirildi. To'g'ri yechim
— M2 dan blok koordinatalarini olish; M3 doirasida emas.
Dalil: L-12, L-13, L-14, L-15.

### Kod skeleti
```python
import re, unicodedata

def normalize_unicode(t: str) -> str:
    return unicodedata.normalize("NFKC", t)      # ligatura: ﬁ → fi

def build_vocabulary(pages) -> frozenset[str]:
    """Korpusdan tasdiqlangan so'zlar — `bookshelves` ni oldini oladi."""
    return frozenset(t for p in pages for t in _TOKEN_RE.findall(normalize_unicode(p).lower()))

def fix_hyphenation(t: str, vocabulary=None) -> str:
    # `+` majburiy: `(\w)-` faqat oxirgi belgini ushlaydi
    if vocabulary is None:
        return re.sub(r"(\w+)-\n[ \t]*(\w+)", r"\1\2", t)
    return re.sub(r"(\w+)-\n[ \t]*(\w+)",
                  lambda m: m.group(1) + m.group(2)
                  if (m.group(1) + m.group(2)).lower() in vocabulary else m.group(0),
                  t)

def normalize_whitespace(t: str) -> str:
    t = t.replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)       # paragraf chegaralarini saqla!
    return t.strip()
```

> Paragraf chegarasini (`\n\n`) **yo'qotma** — M4 chunking shunga tayanadi.

> Adabiyot PDF'lari **mazmun** jihatidan haqiqiy (Gutenberg matni),
> lekin **artefakt** jihatidan sun'iy — MuPDF avtomatik hyphenation
> qilmaydi va header qo'yilmaydi. Artefakt dalili faqat tashqi
> arXiv/PLOS PDF'laridan olinadi.

### Qanday tekshirasan
- "Oldin/keyin" juftliklarini ko'z bilan solishtir (5 ta sahifa).
- `AI`, `API` kabi atamalar saqlanib qolgan.
- Funksiya idempotent: `clean(clean(x)) == clean(x)`.

### Definition of Done
Xom matn izchil ML-tayyor matnga aylanadi; 3 strategiya solishtirilgan va tanlov asoslangan; testlar bor.

**Commit:** `M3: text preprocessing and strategy comparison`

---

## M4 — Chunking

**Maqsad:** hujjatni tahlil qilinadigan birliklarga bo'lish.

```text
✅ PDF → ~100 chunk → 100 vektor → cluster'lar
❌ PDF → 1 vektor
```

**Qayerda:** `notebooks/03_chunking.ipynb` → `app/nlp/chunking.py` + `Chunk` modeli `app/schemas/document.py` ga.

### O'rganish
Chunk o'lchami trade-off'i: juda kichik → ma'no yo'q, juda katta → bir nechta mavzu aralashadi.
Variantlar: sahifa / paragraf / gap guruhlari / fixed-window / semantik. **Boshlash:** `Paragraph → Chunk`.

### Chunk modeli
```python
class Chunk(BaseModel):
    id: int
    text: str           # display matni
    pages: list[int]    # qaysi sahifa(lar)dan
    n_chars: int
```

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M4.T1** | Notebook: parametrlar solishtirish (200/800, 300/1200, 500/2000) | Experiments | M3 | `notebooks/03_chunking.ipynb` |
| **M4.T2** | `Chunk` sxemasi + `chunking.py` (merge/split, sahifa tracking) | NLP | M4.T1 | `app/schemas/document.py`, `app/nlp/chunking.py` |
| **M4.T3** | Tanlangan parametrlarni `config.py` ga ko'chirish | Config | M4.T2 | `app/core/config.py` |
| **M4.T4** | Unit testlar (chegara holatlari, sahifa tracking) | Tests | M4.T3 | `tests/test_chunking.py` |

**Muhim:** `pages` bo'sh bo'lmasligi shart (I/O contract §2.1). Chunk < 10 → `NotEnoughContentError`.
### Qadamlar
- [ ] Sahifalarni birlashtir, paragraflarga bo'l (`\n\n`).
- [ ] **Qisqa paragraflarni birlashtir** (< `MIN_CHARS`, masalan 300) keyingisi bilan.
- [ ] **Uzun paragraflarni bo'l** (> `MAX_CHARS`, masalan 1200) gap chegarasida.
- [ ] Har chunk'ga **sahifa raqami(lar)ini** biriktir (keyin "muhim bo'limlar" uchun kerak).
- [ ] Juda qisqa chunk'larni (sarlavha, raqam) tashla yoki qo'shni bilan birlashtir.
- [ ] Chunk statistikasini chiqar: soni, o'rtacha/min/max uzunlik, histogramma.
- [ ] 10 ta tasodifiy chunk'ni o'qi: "bu mustaqil, ma'noli birlikmi?"
- [ ] Parametrlarni solishtir: `(200, 800)` vs `(300, 1200)` vs `(500, 2000)` — chunk soni va ko'rinishi qanday o'zgaradi?
- [ ] Tanlangan parametrlarni `app/core/config.py` ga ko'chir.

### Qoida
- Maqsad chunk soni: kichik PDF uchun ~20–50, o'rta uchun ~80–300. Agar < 10 bo'lsa — clustering ma'nosiz (`NotEnoughContentError`).

### Definition of Done
Har qanday matnli PDF ma'noli chunk'lar ro'yxatiga aylanadi, har biri sahifa ma'lumotiga ega.

**Commit:** `M4: paragraph-based chunking with page tracking`

---

## M5 — TF-IDF

**Maqsad:** matnni sonli vektorga aylantirish.
**Qayerda:** `notebooks/04_tfidf_manual_and_sklearn.ipynb` → `app/ml/vectorizers.py`.

### O'rganish (tartib bilan)
1. Bag of Words → 2. Vocabulary → 3. Term Frequency (TF) → 4. Inverse Document Frequency (IDF) → 5. TF-IDF → 6. Sparse matritsalar.

**Formula (intuitsiya):**
```text
TF(t, d)  = t so'zining d hujjatdagi soni / d dagi jami so'zlar
IDF(t)    = log( N / df(t) )          # df — t uchraydigan hujjatlar soni
TF-IDF    = TF × IDF
```
Ya'ni: *bu hujjatda ko'p uchraydi, lekin hamma joyda uchramaydi* → muhim so'z.
(Eslatma: sklearn `smooth_idf=True` va L2-normalizatsiya ishlatadi, shuning uchun qo'lda hisoblagan qiymating sklearn'dan **biroz farq qiladi** — bu normal; farq sababini tushun.)

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M5.T1** | Notebook: qo'lda TF-IDF → NumPy → sklearn | Experiments | M4 | `notebooks/04_tfidf_manual_and_sklearn.ipynb` |
| **M5.T2** | `build_tfidf()` — parametrlar expose qilingan | ML | M5.T1 | `app/ml/vectorizers.py` |
| **M5.T3** | `stop_words` parametrini tilga bog'liq qilish | ML | M5.T2 | `app/ml/vectorizers.py` |
| **M5.T4** | Unit testlar (shape, sparsity, empty vocab) | Tests | M5.T3 | `tests/test_vectorizers.py` |

**Muhim (M-T bilan bog'liq):** `stop_words` **config dan** olinadi — `translation_enabled=false` → `None` (ko'p tilli), `true` → `"english"`. Hardcode `"english"` **yo'q**.
### Qadamlar
- [ ] 4 ta qisqa gapdan iborat mini-korpus yarat. **Qo'lda** (qog'ozda) vocabulary, TF, IDF, TF-IDF hisobla.
- [ ] Shuni NumPy bilan kodla (smooth'siz, normalizatsiyasiz).
- [ ] `TfidfVectorizer` bilan qil; natijani solishtir, farqni tushuntir.
- [ ] `vectorizer.get_feature_names_out()` — vocabulary'ni ko'r.
- [ ] `X.shape`, `X.nnz`, sparsity (nol foizi).
- [ ] Parametrlar bilan tajriba (har birining ta'sirini yoz):

  | Parametr | Nima qiladi | Sinab ko'r |
  |----------|-------------|------------|
  | `stop_words` | umumiy so'zlarni olib tashlaydi | `None` vs `"english"` |
  | `min_df` | kam uchraydigan so'zlarni tashlaydi | `1`, `2`, `3` |
  | `max_df` | juda keng tarqalgan so'zlarni tashlaydi | `0.8`, `0.95` |
  | `ngram_range` | so'z juftliklari | `(1,1)` vs `(1,2)` |
  | `max_features` | vocabulary chegarasi | `None`, `5000` |
  | `sublinear_tf` | TF'ga log qo'llaydi | `False` vs `True` |
- [ ] Real PDF chunk'lariga qo'lla; eng yuqori TF-IDF so'zlarini bitta chunk uchun chiqar — ma'noli-mi?
- [ ] `app/ml/vectorizers.py` ga `build_tfidf(chunks, **params)` yoz (vectorizer va matritsa qaytaradi).

### Kod skeleti
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(texts, *, min_df=2, max_df=0.85, ngram_range=(1, 2),
                stop_words="english", sublinear_tf=True):
    vec = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range,
                          stop_words=stop_words, sublinear_tf=sublinear_tf)
    X = vec.fit_transform(texts)
    return vec, X
```

### Tipik xatolar
- Kichik hujjatda `min_df=2` deyarli hamma so'zni o'chiradi → vocabulary bo'sh (`ValueError: empty vocabulary`).
- `fit` ni test ma'lumotida ham chaqirish (data leakage) — bu yerda hamma chunk'lar bir hujjatdan, shuning uchun `fit_transform` OK, lekin farqini bil.

### Definition of Done
Chunk'lar TF-IDF matritsasi sifatida ifodalangan; qo'lda hisob = NumPy = sklearn (farqlar tushuntirilgan).

**Commit:** `M5: tfidf manual + sklearn vectorizer`

---

## M6 — K-Means

**Maqsad:** yorliqsiz, o'xshash chunk'larni guruhlash.
**Qayerda:** `notebooks/05_kmeans_manual_and_sklearn.ipynb` → `app/ml/clustering.py`.

```text
Kirish:  Vektor 1 … Vektor N
Chiqish: Cluster 0, Cluster 1, Cluster 2, …
```

### O'rganish
Centroid, masofa (Euclidean / cosine), assignment, iteratsiya, convergence, `k`, inertia.

**Algoritm:**
```text
1. k ta boshlang'ich centroid tanla
2. Har nuqtani eng yaqin centroid'ga tayinla
3. Har cluster'ning markazini qayta hisobla (o'rtacha)
4. Tayinlashlar o'zgarmay qolguncha 2–3 ni takrorla
Inertia = Σ (nuqta − o'z centroid'i)²   → kichikroq = zichroq cluster'lar
```

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M6.T1** | Notebook: 2D qo'lda 2 iteratsiya → NumPy → sklearn | Experiments | M5 | `notebooks/05_kmeans_manual_and_sklearn.ipynb` |
| **M6.T2** | `cluster_kmeans()` (deterministik, `random_state=42`) | ML | M6.T1 | `app/ml/clustering.py` |
| **M6.T3** | `top_terms()` — centroid'dan eng katta TF-IDF | ML | M6.T2 | `app/ml/clustering.py` |
| **M6.T4** | `representative_chunks()` — centroid'ga eng yaqin 3 ta | ML | M6.T3 | `app/ml/clustering.py` |
| **M6.T5** | Unit testlar (determinism, o'lcham, k <= n) | Tests | M6.T4 | `tests/test_clustering.py` |

**Muhim (M-T bilan bog'liq):** `top_terms` faqat **TF-IDF** matritsasidan olinadi — embedding'da so'z yo'q (M10 ga qarang).
### Qadamlar
- [ ] 2D'da 8–10 ta nuqta bilan **qo'lda** 2 iteratsiya hisobla.
- [ ] NumPy bilan KMeans yoz (~25 qator), 2D sintetik ma'lumotda sina, matplotlib'da chiz.
- [ ] `sklearn.cluster.KMeans` bilan solishtir (`n_init`, `random_state=42`).
- [ ] Nega boshlang'ich centroid'lar muhim? (k-means++ ni o'qi.)
- [ ] TF-IDF chunk'lariga qo'lla; `k = 3, 4, 5, 6, 8` ni sinab ko'r.
- [ ] Cluster o'lchamlarini ko'r (`np.bincount(labels)`) — bitta cluster 80% bo'lsa, bu signal.
- [ ] **Representative chunk'lar:** har cluster uchun centroid'ga eng yaqin 3 ta chunk.
- [ ] **Muhim terminlar:** har cluster centroid'idagi eng katta TF-IDF vaznli 8–10 so'z.
- [ ] Natijani pandas jadvaliga yig'.
- [ ] `app/ml/clustering.py` ga `cluster_kmeans(X, k)` yoz.

### Kod skeleti
```python
import numpy as np
from sklearn.cluster import KMeans

def cluster_kmeans(X, k: int, seed: int = 42):
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    labels = km.fit_predict(X)
    return km, labels

def top_terms(km, feature_names, n=10):
    order = km.cluster_centers_.argsort(axis=1)[:, ::-1]
    return {c: [feature_names[i] for i in order[c, :n]] for c in range(km.n_clusters)}

def representative_chunks(X, km, labels, n=3):
    from sklearn.metrics import pairwise_distances
    reps = {}
    for c in range(km.n_clusters):
        idx = np.where(labels == c)[0]
        d = pairwise_distances(X[idx], km.cluster_centers_[c].reshape(1, -1)).ravel()
        reps[c] = idx[np.argsort(d)[:n]].tolist()
    return reps
```

### Qanday tekshirasan
- Har cluster'ning 3 representative chunk'ini o'qi: ular rostdan bir mavzumi?
- Top terms ma'noli-mi, yoki "the, of, and" turibdimi (u holda M3/M5 parametrlariga qayt)?

### Tipik xatolar
- Boshqa `random_state` → boshqa natija. Doim qotir.
- Cluster soni chunk sonidan katta bo'lib ketishi.

### Definition of Done
PDF chunk'lari cluster'larga ajratilgan; har cluster uchun o'lcham, representative chunk'lar va top terms bor.

**Commit:** `M6: kmeans manual + sklearn, representative chunks, top terms`

---

## M7 — Cluster baholash

**Maqsad:** cluster'lar foydali-yo'qligini o'lchash va `k` ni asoslab tanlash.
**Qayerda:** `notebooks/06_cluster_evaluation.ipynb` → `app/ml/evaluation.py`.

### Metrikalar
| Metrika | Ma'nosi | Yaxshi qiymat |
|---------|---------|---------------|
| Inertia | cluster ichidagi kvadrat masofalar yig'indisi | kichik (lekin `k` oshsa doim kamayadi!) |
| Silhouette | `(b − a) / max(a, b)`; `a` — o'z cluster'iga, `b` — eng yaqin boshqa cluster'ga o'rtacha masofa | −1…1, yuqori yaxshi |
| Cluster o'lchami | har cluster'dagi chunk soni | muvozanatli, bo'sh/yagona-chunk'li yo'q |
| Intra-cluster similarity | cluster ichidagi o'rtacha cosine | yuqori |
| Inter-cluster separation | centroid'lar orasidagi masofa | yuqori |

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M7.T1** | Notebook: elbow + silhouette + stability (5 seed) | Experiments | M6 | `notebooks/06_cluster_evaluation.ipynb` |
| **M7.T2** | `evaluate_k_range()` — jadval qaytarish | ML | M7.T1 | `app/ml/evaluation.py` |
| **M7.T3** | `suggest_k()` — silhouette eng yuqori `k` (heuristic) | ML | M7.T2 | `app/ml/evaluation.py` |
| **M7.T4** | Qarorni yozish: "k=... ni tanladim, chunki ..." | Docs | M7.T3 | `docs/learning_notes.md` |

**Muhim:** `suggest_k` — **tavsiya**, xulosa emas. Yakuniy qaror = raqam + qo'lda ko'rish.
### Qadamlar
- [ ] `k = 2…10` (chunk soniga qarab) uchun inertia hisobla, **elbow** grafigini chiz.
- [ ] Har `k` uchun `silhouette_score(X, labels, metric="cosine")`.
- [ ] Jadval yig' (`k`, inertia, silhouette, min/max cluster o'lchami).
- [ ] Har nomzod `k` uchun cluster'larni **qo'lda** ko'r (M6 dagi top terms + representative chunk'lar).
- [ ] `random_state` ni 5 xil o'zgartirib, natija barqarormi tekshir (stability).
- [ ] Qaror yoz: "`k=…` ni tanladim, chunki …" (raqam + qo'lda kuzatuv).
- [ ] Avtomatik `k` tanlash funksiyasi: silhouette eng yuqori bo'lgan `k` ni **tavsiya** qiladi (lekin yagona haqiqat emas).
- [ ] Cheklovlarni yoz: silhouette matnli sparse ma'lumotda past bo'lishi tabiiy (0.02–0.15 ham odatiy).

### Kod skeleti
```python
from sklearn.metrics import silhouette_score

def evaluate_k_range(X, k_values, seed=42):
    rows = []
    for k in k_values:
        km, labels = cluster_kmeans(X, k, seed)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels, metric="cosine"),
            "min_size": int(np.bincount(labels).min()),
            "max_size": int(np.bincount(labels).max()),
        })
    return rows
```

> **Muhim:** bitta "to'g'ri" `k` yo'q. **Miqdoriy metrika + sifat bo'yicha ko'z bilan tekshiruv** — ikkalasi birga.

### Definition of Done
Tanlangan konfiguratsiya o'lchovlar va qo'lda tekshiruv bilan asoslangan; cheklovlar yozilgan.

**Commit:** `M7: cluster evaluation (elbow, silhouette, stability)`

---

## M8 — PCA vizualizatsiya

**Maqsad:** yuqori o'lchamli vektorlarni 2D'da ko'rish.
**Qayerda:** `notebooks/07_pca_visualization.ipynb` → `app/ml/visualization.py`.

### O'rganish
O'lcham (dimensionality), PCA, dispersiya (variance), explained variance, 2D proyeksiya.

> ⚠️ `PCA` sklearn'da **sparse matritsani qabul qilmaydi**. TF-IDF uchun `TruncatedSVD(n_components=2)` ishlat (yoki `X.toarray()` — faqat kichik hujjatda). Embedding'lar (dense) bilan oddiy `PCA` ishlaydi.

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M8.T1** | Notebook: TruncatedSVD (sparse) va PCA (dense) | Experiments | M7 | `notebooks/07_pca_visualization.ipynb` |
| **M8.T2** | `plot_clusters_2d()` + explained variance izohi | Viz | M8.T1 | `app/ml/visualization.py` |

**Muhim:** TF-IDF **sparse** → `PCA` ishlamaydi, `TruncatedSVD` kerak. 2D grafik — **illyustratsiya, isbot emas**.
### Qadamlar
- [ ] 3D → 2D ning kichik qo'lda misolini tushun (qaysi yo'nalishda dispersiya katta).
- [ ] TF-IDF'ga `TruncatedSVD(2)` qo'lla.
- [ ] Scatter plot: rang = cluster label.
- [ ] `explained_variance_ratio_` ni chiqar va izohla (2D odatda dispersiyaning oz qismini oladi — bu normal).
- [ ] Cluster'lar ajralganmi, ustma-ustmi? Natijani yoz.
- [ ] Ikki o'lchamli rasm **isbot emas**, faqat illyustratsiya ekanini yoz (3D/ko'p o'lchamdagi ajralish 2D'da yo'qolishi mumkin).
- [ ] (Ixtiyoriy, keyin) t-SNE / UMAP.
- [ ] Grafikni `docs/figures/` ga saqla; `plot_clusters_2d(X, labels)` ni modulga ko'chir.

### Definition of Done
Cluster tuzilmasi 2D'da ko'rsatilgan, explained variance izohlangan.

**Commit:** `M8: 2D cluster visualization (SVD/PCA)`

---

## M9 — Semantik embedding'lar

**Maqsad:** so'z chastotasi emas, **ma'noni** ifodalash (`car` ≈ `automobile`).
**Qayerda:** `notebooks/08_embeddings.ipynb` → `app/ml/vectorizers.py` ga `build_embeddings()`.

### O'rganish
- Embedding nima, o'lchami (masalan 384), cosine similarity:
  ```text
  cos(a, b) = (a · b) / (‖a‖ ‖b‖)
  ```
- TF-IDF'ning kamchiligi: sinonimlarni bilmaydi.

### Modellar
| Model | Til | Izoh |
|-------|-----|------|
| `all-MiniLM-L6-v2` | asosan ingliz | tez, kichik — boshlash uchun |
| `paraphrase-multilingual-MiniLM-L12-v2` | ko'p tilli | boshqa tillar (o'zbek sifati cheklangan bo'lishi mumkin — sinab ko'r) |

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M9.T1** | Notebook: embedding + cosine + `max_seq_length` tekshiruvi | Experiments | M8 | `notebooks/08_embeddings.ipynb` |
| **M9.T2** | `build_embeddings()` — global model cache, `normalize=True` | ML | M9.T1 | `app/ml/vectorizers.py` |
| **M9.T3** | **Ko'p tilli** rejim (`USE_MULTILINGUAL`) | ML | M9.T2 | `app/ml/vectorizers.py` |
| **M9.T4** | TF-IDF vs Embedding solishtirma jadvali | Docs | M9.T3 | `notebooks/08_embeddings.ipynb` |
| **M9.T5** | Unit testlar (shape, cosine ∈ [-1,1]) | Tests | M9.T4 | `tests/test_vectorizers.py` |

**Muhim (M-T bilan bog'liq):**
- `USE_MULTILINGUAL=true` → `paraphrase-multilingual-MiniLM-L12-v2` (tarjima **yo'q** holat)
- `USE_MULTILINGUAL=false` → `all-MiniLM-L6-v2` (tarjima **yoqilgan** holat)
- Embedding uchun **`display_text`** ishlatiladi (stemming'li matn zarar qiladi)
- `model.max_seq_length` (odatda 256) → uzun chunk'lar kesiladi, chunk o'lchamiga ta'siri yoziladi
### Qadamlar
- [ ] Model yukla: `SentenceTransformer("all-MiniLM-L6-v2")` (birinchi marta internetdan yuklanadi).
- [ ] 5–6 ta gapni kodla, `shape` ni ko'r.
- [ ] Cosine similarity matritsasi: sinonim/o'xshash gaplar yuqori bahoga ega ekanini ko'r.
- [ ] Qo'lda cosine'ni NumPy bilan hisobla, `sklearn.metrics.pairwise.cosine_similarity` bilan solishtir.
- [ ] Real PDF chunk'larini kodla: `normalize_embeddings=True`.
- [ ] **Muhim:** embedding uchun **`display_text`** (yengil tozalangan) ishlat — stop-word'siz/stemming'li matn embedding modeliga zarar qiladi.
- [ ] Modelning maksimal token chegarasini (`model.max_seq_length`, odatda 256) tekshir — uzun chunk'lar kesiladi! Chunk o'lchamiga ta'sirini yoz.
- [ ] Embedding'larni `data/processed/` ga keshla (`np.save`) — har safar qayta hisoblama.
- [ ] K-Means'ni embedding'larda ishlat (normalizatsiya qilingan vektorlarda Euclidean ≈ cosine).
- [ ] **TF-IDF vs Embedding** solishtirma jadvali (silhouette, cluster o'lchamlari, qo'lda ko'rilgan ma'nolilik).
- [ ] `build_embeddings(texts, model_name)` ni modulga yoz; modelni **bir marta** yukla (global/cache).

### Kod skeleti
```python
from sentence_transformers import SentenceTransformer
import numpy as np

_model = None
def build_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
```

### Definition of Done
Loyiha semantik vektorlarni qo'llaydi; TF-IDF va embedding solishtirilgan; vektorlar keshlanadi.

**Commit:** `M9: sentence-transformer embeddings + comparison`

---

## M10 — Topic Discovery

**Maqsad:** anonim cluster'larni **talqin qilinadigan mavzularga** aylantirish.
**Qayerda:** `notebooks/09_topic_discovery.ipynb` → `app/ml/topics.py`.

```text
Cluster → Representative chunk'lar → Muhim so'zlar → Topic nomzodi
```
K-Means faqat "Cluster 0/1/2" beradi; ma'no berish — bu bosqich.

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M10.T1** | Notebook: main topic qoidasi (size vs centroid) | Experiments | M9 | `notebooks/09_topic_discovery.ipynb` |
| **M10.T2** | `Topic` sxemasi | Schemas | M10.T1 | `app/schemas/analysis.py` |
| **M10.T3** | `discover_topics()` — size, share, pages | ML | M10.T2 | `app/ml/topics.py` |
| **M10.T4** | Embedding rejimi uchun **top terms** (TF-IDF orqali) | ML | M10.T3 | `app/ml/topics.py` |
| **M10.T5** | Representative chunk'lar (embedding centroid'iga cosine) | ML | M10.T4 | `app/ml/topics.py` |
| **M10.T6** | Unit testlar (share summasi = 1, pages unique/sorted) | Tests | M10.T5 | `tests/test_topics.py` |

**Muhim:** embedding centroid'i **so'z bermaydi** — top terms uchun cluster chunk'lari TF-IDF dan o'tkaziladi.
### Qadamlar
- [ ] Topic ma'lumot modeli (Pydantic/dataclass):
  ```python
  class Topic(BaseModel):
      cluster_id: int
      size: int
      share: float                 # chunk'larning ulushi
      top_terms: list[str]
      representative_chunks: list[int]   # chunk id'lar
      pages: list[int]             # cluster qamrab olgan sahifalar
      label: str | None = None     # LLM (M11) to'ldiradi
      description: str | None = None
  ```
- [ ] Embedding ishlatilganda ham **top terms** ni olish usuli: cluster chunk'larini TF-IDF'dan o'tkaz va cluster ichidagi o'rtacha vaznli so'zlarni ol (embedding centroid'i so'z bermaydi).
- [ ] Representative chunk'larni embedding centroid'iga cosine bo'yicha tanla.
- [ ] Har cluster uchun sahifalar oralig'ini hisobla (muhim bo'limlar shundan chiqadi).
- [ ] **Main topic** qoidasi: eng katta cluster YOKI hujjat markaziga (barcha embedding'lar o'rtachasiga) eng yaqin cluster — ikkalasini sinab ko'r, qaysi biri ma'noliroq ekanini yoz.
- [ ] **Muhim bo'limlar:** har cluster'ning representative chunk'lari + sahifa raqamlari.
- [ ] Qo'lda, vaqtincha nom ber (`Cluster 0 → Neural Networks`) va top terms bilan moslikni tekshir.
- [ ] `discover_topics(chunks, X_for_terms, X_for_clustering, labels)` funksiyasi `list[Topic]` qaytarsin.

### Definition of Done
Har cluster'da: representative chunk'lar, muhim so'zlar, sahifalar, o'lcham va talqin qilinadigan tavsif nomzodi bor.

**Commit:** `M10: topic objects from clusters`

---

## M11 — Ollama integratsiyasi

**Maqsad:** lokal LLM'ni **tushuntirish qatlami** sifatida ulash.
**Qayerda:** `app/llm/ollama_client.py`, `app/llm/prompts.py`, `tests/test_ollama_client.py`.

```text
Python → Ollama HTTP API (localhost:11434) → lokal model
```

> LLM nima qiladi: mavzuga nom beradi, cluster'ni tushuntiradi, key points chiqaradi, yakuniy xulosa yozadi. LLM nima **qilmaydi**: cluster'lar/mavzularni o'zi topmaydi.

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M11.T1** | Notebook: Ollama API ni tekshirish (`/api/tags`, `/api/generate`) | Experiments | M10 | `notebooks/` (tez-tez) |
| **M11.T2** | `OllamaUnavailableError` + `base.py` (LLM interface) | Core | M11.T1 | `app/core/exceptions.py`, `app/llm/base.py` |
| **M11.T3** | `OllamaClient` — timeout, xato, retry | LLM | M11.T2 | `app/llm/ollama_client.py` |
| **M11.T4** | `prompts.py` — 3 prompt (label, key_points, summary) | LLM | M11.T3 | `app/llm/prompts.py` |
| **M11.T5** | `generate_json()` — `format=json` + Pydantic validatsiya | LLM | M11.T4 | `app/llm/ollama_client.py` |
| **M11.T6** | Unit testlar (mock) + `@pytest.mark.integration` | Tests | M11.T5 | `tests/test_ollama_client.py` |

**Muhim:** `temperature=0`, retry 1–2 marta, kontekst chegarasi (~500–800 belgi/chunk), "faqat berilgan matnga tayan" talabi.
### Qadamlar
- [ ] `/api/generate` yoki `/api/chat` ga oddiy so'rov (`stream=False`).
- [ ] Timeout, ulanish xatosi, model topilmadi holatlari uchun xato ushlash (`OllamaUnavailableError`).
- [ ] **Strukturalangan chiqish:** `"format": "json"` + prompt'da aniq sxema; javobni Pydantic bilan validatsiya qil; yaroqsiz bo'lsa 1–2 marta qayta urin.
- [ ] Prompt'lar (`prompts.py` da alohida konstantalar):
  1. **Topic label:** top terms + 3 representative chunk (qisqartirilgan) → `{"label": "...", "description": "..."}`
  2. **Key points:** barcha topic'lar tavsifi → `{"key_points": [...]}`
  3. **Final summary:** hamma topic'lar → `{"main_topic": "...", "summary": "..."}`
- [ ] **Kontekst chegarasi:** LLM'ga butun PDF'ni yubormа! Faqat topic'lar bo'yicha qisqa kontekst (har chunk ~500–800 belgigacha kes).
- [ ] Prompt'da talab qil: "Faqat berilgan matnga tayan, o'ylab topma."
- [ ] Natija tilini boshqarish (`output_language` parametri).
- [ ] Test: Ollama o'chiq bo'lsa tushunarli xato; mock bilan unit-test.

### Kod skeleti
```python
import httpx, json
from app.core.config import settings

class OllamaClient:
    def __init__(self, base_url=None, model=None, timeout=120):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout

    def generate_json(self, prompt: str) -> dict:
        r = httpx.post(f"{self.base_url}/api/generate",
                       json={"model": self.model, "prompt": prompt,
                             "stream": False, "format": "json"},
                       timeout=self.timeout)
        r.raise_for_status()
        return json.loads(r.json()["response"])
```

### Tipik xatolar
- Kichik model JSON'ni buzadi → qayta urinish + sodda sxema.
- Prompt juda uzun → javob sifati tushadi/timeout.
- Docker ichidan `localhost` Ollama'ni ko'rmaydi (M16 ga qara).

### Definition of Done
Python lokal Ollama bilan gaplashadi va **validatsiyalangan strukturali** javob oladi.

**Commit:** `M11: ollama client with structured json output`

---

## M12 — To'liq tahlil pipeline'i

**Maqsad:** hamma komponentni bitta servisga bog'lash.
**Qayerda:** `app/services/analysis_service.py`, `tests/test_pipeline.py`, tajriba — kerak bo'lsa `notebooks/`.

```text
PDF → Extract → Clean → Chunk → Vectorize (TF-IDF/Embedding) → Cluster
    → Evaluate → Topics → Ollama → Final result
```

### Natija formati
```text
Document
├── Main Topic
├── Discovered Topics (+ tushuntirish)
├── Key Points
├── Important Sections (sahifa raqamlari bilan)
└── Overall Summary
```

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M12.T1** | `AnalysisResult` sxemasi (I/O contract §2.2) | Schemas | M11 | `app/schemas/analysis.py` |
| **M12.T2** | `analysis_service.py` — pipeline tartibi | Service | M12.T1 | `app/services/analysis_service.py` |
| **M12.T3** | **Translation integratsiyasi** (ixtiyoriy, pre-chunk) | Service | M-T | `app/services/analysis_service.py` |
| **M12.T4** | Avtomatik `k` tanlash + `NotEnoughContentError` | Service | M12.T2 | `app/services/analysis_service.py` |
| **M12.T5** | Timing (`perf_counter`) + logging | Service | M12.T4 | `app/services/analysis_service.py` |
| **M12.T6** | Graceful degradation (LLM yo'q → ML natijasi) | Service | M12.T5 | `app/services/analysis_service.py` |
| **M12.T7** | E2E testlar (mock LLM, translation yoqiq/yoqilgan) | Tests | M12.T6 | `tests/test_pipeline.py` |

**Pipeline tartibi (majburiy):**
```text
extract_pages → [OPTIONAL translate: document/page] → preprocess
→ chunk → vectorize → cluster → evaluate → topics → LLM → AnalysisResult
```

**Xato modeli (system-architect §19 — ichki detallar sizib chiqmaydi):**

```json
{
  "error": {
    "code": "TRANSLATION_UNAVAILABLE",
    "message": "Translation provider unavailable",
    "request_id": "..."
  }
}
```

> **Translation M12.T3 da** — chunk'lashdan **OLDIN**. `topics` hech qachon bo'sh qolmaydi; LLM maydonlari `None` bo'lishi mumkin.
### Qadamlar
- [ ] `analyze_pdf(source, *, representation="embedding", k=None) -> AnalysisResult` yoz.
- [ ] `k=None` bo'lsa avtomatik tanlash (M7 funksiyasi), chunk soniga qarab chegaralab (`k_max = min(10, n_chunks // 5)`).
- [ ] Chunk soni juda kam bo'lsa → `NotEnoughContentError`.
- [ ] Har bosqichni **loglа** va **vaqtini o'lcha** (`time.perf_counter`) — qaysi bosqich sekin?
- [ ] LLM yiqilsa: ML natijasi (topic'lar, top terms) baribir qaytsin, LLM maydonlari `None` bo'lsin (graceful degradation).
- [ ] 3 ta namuna PDF'da to'liq ishga tushir va natijalarni qo'lda o'qi.
- [ ] `AnalysisResult` Pydantic modeli (M13 API javobiga asos).

### Definition of Done
Bitta funksiya PDF'ni boshidan oxirigacha tahlil qiladi va to'liq strukturali natija qaytaradi.

**Commit:** `M12: end-to-end analysis pipeline`

---

## M13 — FastAPI

**Maqsad:** pipeline'ni HTTP orqali ochish.
**Qayerda:** `app/main.py`, `app/api/routes_*.py`, `app/schemas/analysis.py`, `tests/test_api.py`.

### Endpoint'lar
```http
POST /documents/analyze      # asosiy: PDF → to'liq tahlil
GET  /health                 # tirikligini tekshirish (+ Ollama holati)
POST /documents/upload       # keyingi bosqich: saqlash
GET  /documents/{id}         # keyingi bosqich: natijani olish
```

### Javob namunasi
```json
{
  "document": "paper.pdf",
  "main_topic": "...",
  "topics": [],
  "clusters": [],
  "summary": "...",
  "key_points": []
}
```

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M13.T1** | FastAPI app + `lifespan` (model bir marta) | API | M12 | `app/main.py` |
| **M13.T2** | Fayl validatsiyasi (magic bytes, hajm, content_type, filename sanitize) | API | M13.T1 | `app/api/routes_documents.py` |
| **M13.T3** | `POST /documents/analyze` (threadpool, `def`) | API | M13.T2 | `app/api/routes_documents.py` |
| **M13.T4** | `GET /health` (+ Ollama holati) | API | M13.T1 | `app/api/routes_health.py` |
| **M13.T5** | Xato xaritasi (400/413/422/503/500) + **Translation → 503** | API | M13.T3 | `app/main.py`, `app/api/` |
| **M13.T6** | CORS aniq + rate limit (limit/window/scope) | Security | M13.T5 | `app/main.py` |
| **M13.T7** | API testlari (`TestClient`, barcha status kodlar) | Tests | M13.T6 | `tests/test_api.py` |

**Endpoint contract (9 maydon — backend §20):**

| Maydon | `/documents/analyze` | `/health` |
|--------|--------------------|-----------|
| Method | `POST` | `GET` |
| Authn | Yo'q (MVP, lokal) | Yo'q |
| Request | `multipart/form-data`, `file` | — |
| Validation | magic bytes `%PDF`, ≤ `MAX_UPLOAD_MB`, `needs_pass` | — |
| Response | `AnalysisResult` | `status`, `version`, `ollama` |
| Status | 200, 400, 413, 422, 503, 500 | 200, 503 |
| Error | `{error: {code, message, request_id}}` (system-arch §19) | Xuddi shu |
| Idempotency | Tekislik, state-siz | O'zgarmas |

**Xavfsizlik (`security §25` — fayl upload majburiy):**
- `file size` ✅ (`MAX_UPLOAD_MB`) · `file type` ✅ (magic bytes) · `MIME validation` ✅
- `filename sanitization` — **kerak** · `path traversal` (`../`) — **kerak**
- `storage isolation` — vaqtinchalik fayl, `M12` tugagach o'chiriladi

> **"Never trust the client-provided MIME type or filename."** (`security §25`)
> **"Frontend validation is not a security boundary."** (`security §22`)

**Muhim:** og'ir hisob **event loop'ni bloklamasligi** uchun endpoint `def` bo'lishi shart (FastAPI threadpool ishlatadi).
> `Allow-Origin: *` **yo'q** — CORS aniq bo'lishi shart (`security §29`).
### Qadamlar
- [ ] FastAPI app + routerlar (`include_router`).
- [ ] `POST /documents/analyze` — `UploadFile` qabul qilsin (`python-multipart` kerak).
- [ ] **Fayl validatsiyasi:** kengaytma `.pdf`, `content_type`, hajm chegarasi (`MAX_UPLOAD_MB`), bo'sh fayl, PDF "magic bytes" (`%PDF`).
- [ ] Servis chaqiruvi sinxron og'ir hisob — event loop'ni bloklamaslik uchun `def` endpoint (FastAPI threadpool'da ishlatadi) yoki `run_in_threadpool`.
- [ ] **Xato xaritasi:**

  | Ichki xato | HTTP |
  |-----------|------|
  | Fayl PDF emas / buzilgan | 400 |
  | Hajm katta | 413 |
  | Skanerlangan / kam kontent | 422 |
  | Ollama mavjud emas | 503 (yoki qisman natija) |
  | Kutilmagan xato | 500 (logga yoz, foydalanuvchiga ichki tafsilot ko'rsatma) |
- [ ] Pydantic response modellari (`response_model=`).
- [ ] `GET /health`.
- [ ] Modellarni (embedding) ilova **startup**'ida bir marta yukla (`lifespan`).
- [ ] Swagger'da (`/docs`) qo'lda sina; curl misoli:
  ```bash
  curl -X POST http://localhost:8000/documents/analyze -F "file=@data/raw/paper.pdf"
  ```
- [ ] `TestClient` bilan API testlari.

### Ishga tushirish
```bash
uvicorn app.main:app --reload --port 8000
```

### Definition of Done
To'liq tahlil HTTP orqali ishga tushadi; xatolar to'g'ri status kodlar bilan qaytadi; `/docs` ishlaydi.

**Commit:** `M13: fastapi analyze endpoint with validation`

---

## M14 — Streamlit frontend

**Maqsad:** Postman/curl'siz foydalanish.
**Qayerda:** `frontend/streamlit_app.py`.

```text
PDF Analyzer
────────────────────────
[ Choose PDF ]  [ Analyze ]
────────────────────────
Main Topic · Topics · Clusters · Key Points · Summary
```

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M14.T1** | `api_client.py` — transport + xato normalizatsiyasi + `API_TIMEOUT` | Frontend | M13 | `frontend/api_client.py` |
| **M14.T2** | UI skeleti: uploader + Analyze + spinner | Frontend | M14.T1 | `frontend/streamlit_app.py` |
| **M14.T3** | Natija ko'rsatish (main topic, topics, key points, summary) | Frontend | M14.T2 | `frontend/streamlit_app.py` |
| **M14.T4** | Xatolarni ko'rsatish (4xx/5xx → `st.error`) | Frontend | M14.T3 | `frontend/streamlit_app.py` |
| **M14.T5** | Qo'shimcha: scatter plot + bar chart + `session_state` | Frontend | M14.T4 | `frontend/streamlit_app.py` |

**UI State Model (`frontend §11` — MAJBURIY):**

```text
Initial → Loading → Success
                   → Empty  (topik topilmadi)
                   → Error  (400/413/503/500)
```

> **"Do not design only the happy path."** — `Initial`, `Empty`, `Error` holatlari **shart**.

**Qoidalar:**
- `API_URL` va `API_TIMEOUT` env'dan olinadi — hardcode **yo'q** (devops §18, §58).
- HTTP so'rovlari **faqat** `api_client.py` da — UI faylda yopiq (`frontend §16`).
- Xato: stack trace / ichki API xatosi / DB detail **chiqarilmaydi** (`frontend §13`).
- Tugma so'rov davomida **disabled**, xatodan keyin **qayta yoqiladi** (`frontend §61`).
- `session_state` — faqat UI holati uchun, global store emas (`frontend §21`).
### Qadamlar
- [ ] `st.file_uploader(type=["pdf"])`.
- [ ] "Analyze" tugmasi → `requests.post(API_URL + "/documents/analyze", files=...)`.
- [ ] `st.spinner` (loading holati) — tahlil 10–60+ soniya olishi mumkin.
- [ ] Natija: main topic (`st.header`), topic'lar (`st.expander` har biri uchun: label, top terms, representative chunk'lar, sahifalar), key points ro'yxati, summary.
- [ ] (Yaxshi qo'shimcha) cluster scatter plot (M8) va cluster o'lchamlari bar chart.
- [ ] Xatolarni ko'rsat: API javobi 4xx/5xx → tushunarli `st.error`.
- [ ] API URL ni env'dan ol (`API_URL`), hardcode qilma.
- [ ] `st.session_state` da natijani saqla (har tugma bosilganda qayta hisoblamaslik uchun).

### Ishga tushirish
```bash
streamlit run frontend/streamlit_app.py
```

### Definition of Done
Foydalanuvchi PDF yuklab, tahlilni UI'da oladi; xatolar tushunarli ko'rsatiladi.

**Commit:** `M14: streamlit ui`

---

## M15 — Testlash

**Maqsad:** tizimni ishonchli qilish.
**Qayerda:** `tests/`, `pytest.ini`, `tests/fixtures/`.

### Test qatlamlari
```text
Unit · Integration · API · ML · LLM integration · End-to-End
```

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M15.T1** | Fixture PDF'lar (normal/empty/scanned/1-page/corrupted) | Tests | M2 | `tests/fixtures/` |
| **M15.T2** | `pytest.ini` + `conftest.py` (umumiy fixture'lar) | Tests | M15.T1 | `pytest.ini`, `tests/conftest.py` |
| **M15.T3** | ML property testlari (determinism, label soni, cosine) | Tests | M6–M9 | `tests/test_clustering.py`, `tests/test_vectorizers.py` |
| **M15.T4** | Edge case'lar (9-bo'lim jadvali) | Tests | M13 | `tests/test_*.py` |
| **M15.T5** | E2E (mock LLM) + translation yoqiq/yoqilgan | Tests | M12, M-T | `tests/test_pipeline.py` |
| **M15.T6** | Coverage `pytest --cov=app` (~70%+, **loyiha tanlovi**) | Tests | M15.T5 | — |

**QA workflow (qa-engineer.md §5 — 11 bosqich):**

```text
Understand Requirements → Identify Acceptance Criteria → Risk Analysis
→ Test Plan → Test Design → Implementation → Execute → Defect Analysis
→ Regression → Quality Decision → Report
```

**Test nomlashuvi (`qa §63`):** `extract_pages_returns_empty_for_scanned_pdf` — xulqni tushuntiradi, `test_pdf_1` emas.

**Muhim:** LLM unit testlari **mock** qilinadi. Haqiqiy Ollama uchun `@pytest.mark.integration` (`pytest -m "not integration"` bilan chiqariladi).

> **"A mocked test cannot prove that the real integration works."** (`qa §36`) — Ollama uchun **haqiqiy** smoke test ham kerak (M11.T6, M16.T8).
> **Flaky test** bo'lsa — sababini tekshir, `skip`/`retry` bilan yopma (`qa §37`).
> **Coverage target opencode'da yo'q** (`qa §49`: "High coverage does not guarantee high software quality"). ~70% — bu **loyiha tanlovi**, majburiy emas.
> **Property-based testing** opencode'da majburiy **emas** — M15.T3 property testlari **loyiha tanlovi**.
### Qadamlar
- [ ] `pytest.ini` + `conftest.py` (umumiy fixture'lar: kichik PDF, namunaviy chunk'lar).
- [ ] **Fixture PDF'lar** (`tests/fixtures/`, kichik, <200 KB): normal, bo'sh, skanerlangan (rasm), 1-sahifalik, buzilgan (`b"not a pdf"`).
- [ ] **Unit:** extraction, preprocessing (har funksiya), chunking (chegara holatlari), vectorization (shape), clustering (determinizm), schemas.
- [ ] **ML testlari** (qat'iy raqam emas, **xususiyat** tekshir):
  - bir xil `random_state` → bir xil label'lar
  - label soni = chunk soni
  - `len(set(labels)) <= k`
  - cosine similarity ∈ [−1, 1]; o'zi bilan = 1
  - sintetik, aniq ajralgan 2 ta mavzu matnida K-Means ularni to'g'ri ajratadi
- [ ] **LLM testlari:** Ollama ni **mock** qil (unit); alohida `@pytest.mark.integration` — haqiqiy Ollama (CI'da o'tkazib yuboriladi).
- [ ] **API testlari:** `TestClient`, 200/400/413/422 holatlari.
- [ ] **E2E:** namuna PDF → to'liq natija strukturasi (mock LLM bilan).
- [ ] **Edge case'lar** (9-bo'limdagi jadvalning hammasi uchun test).
- [ ] Qamrov (coverage) ni ko'r: `pytest --cov=app` (`pytest-cov` o'rnat) — maqsad ~70%+ asosiy mantiqda.

### Ishga tushirish
```bash
pytest -q
pytest -m "not integration"
```

### Definition of Done
Asosiy funksionallik avtomatik testlangan, edge case'lar qoplangan, `pytest` yashil.

**Commit:** `M15: test suite (unit, ml, api, e2e)`

---

## M16 — Docker

**Maqsad:** loyihani istalgan joyda bir xil ishga tushirish.

```text
docker-compose
├── backend   (FastAPI + ML)
├── frontend  (Streamlit)
└── ollama    (yoki host'dagi Ollama)
```

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M16.T1** | `.dockerignore` | DevOps | M15 | `.dockerignore` |
| **M16.T2** | Backend `Dockerfile` (Python 3.12, CPU torch, **non-root**, layer cache) | DevOps | M16.T1 | `Dockerfile` |
| **M16.T3** | Frontend image/command | DevOps | M16.T2 | `Dockerfile`, `frontend/` |
| **M16.T4** | `docker-compose.yml` (backend/frontend/ollama) | DevOps | M16.T3 | `docker-compose.yml` |
| **M16.T5** | Ollama wiring (host yoki konteyner) + HF cache volume | DevOps | M16.T4 | `docker-compose.yml` |
| **M16.T6** | **Resource limits** (`mem_limit`, `cpus`) + `healthcheck` | DevOps | M16.T5 | `docker-compose.yml` |
| **M16.T7** | Port exposure policy — Ollama **host'ga publish qilinmaydi** | DevOps | M16.T6 | `docker-compose.yml` |
| **M16.T8** | Toza muhit testi (`down -v` → `up --build`) + health tekshiruvi | Verify | M16.T7 | — |

**Majburiy qoidalar:**

| Qoida | Manba |
|-------|-------|
| `python:3.12-slim` (3.11 emas — ADR-004 bilan mos) | ADR-004 |
| torch CPU-index **alohida layer** (2.5 GB → layer cache) | devops §10 |
| **non-root user** (`USER app`) | devops §10, §11 |
| Runtime image'da compiler **yo'q** | devops §11 |
| `mem_limit` / `cpus` — torch xotira ochad | devops §46 |
| Faqat `8000` va `8501` publish; **Ollama `11434` ichki** | devops §48 |
| `/health` → `healthcheck` (`depends_on` alone ≠ readiness) | devops §25 |
| Config `ENV` bilan image'ga **yomburun emas** — `env_file`/`environment` | devops §9, §19 |

> **Muhim:** `localhost` konteyner ichida **o'zini** bildiradi, host'ni emas. Linux uchun `extra_hosts: ["host.docker.internal:host-gateway"]`.
> Bu **loyiha qoidasi** (opencode hujjatlarida `localhost` yo'q); asos — devops §49 "avoid hardcoding dynamic infrastructure addresses".
### Qadamlar
- [ ] `.dockerignore` (`.venv`, `data/processed`, `__pycache__`, `.git`, `notebooks`).
- [ ] `Dockerfile` (backend): `python:3.11-slim`, avval `requirements.txt` ni nusxala va o'rnat (layer cache), keyin kod.
- [ ] Frontend uchun alohida Dockerfile yoki bitta image + boshqa `command`.
- [ ] `docker-compose.yml`: env o'zgaruvchilar, portlar (8000, 8501), `depends_on`.
- [ ] **Ollama ulanishi:**
  - Ollama **host**'da: `OLLAMA_BASE_URL=http://host.docker.internal:11434` (Linux'da `extra_hosts: ["host.docker.internal:host-gateway"]`).
  - Ollama **konteynerda**: `ollama` servis, `volumes: ollama_data:/root/.ollama` (modellar qayta yuklanmasligi uchun), `OLLAMA_BASE_URL=http://ollama:11434`. Modelni birinchi marta `docker compose exec ollama ollama pull <model>`.
- [ ] **Embedding modeli keshi:** HuggingFace keshini volume'ga ol (`~/.cache/huggingface`) — aks holda har ishga tushishda qayta yuklanadi.
- [ ] `healthcheck` (`/health`).
- [ ] **Toza muhit testi:** `docker compose down -v && docker compose up --build` — noldan ishlaydimi?
- [ ] README'da aniq buyruqlar ketma-ketligini hujjatlashtir.

### Tipik xatolar
- Image hajmi katta (torch) → CPU-only torch yoki slim base; birinchi build uzoq davom etishi normal.
- `localhost` konteyner ichida o'zini bildiradi, host'ni emas.

### Definition of Done
Hujjatlashtirilgan Docker buyruqlari ilovani toza muhitda muvaffaqiyatli ishga tushiradi.

**Commit:** `M16: dockerize backend, frontend, ollama wiring`

---

## M17 — Yakuniy ML baholash

**Maqsad:** haqiqiy **ML tizimini** baholash va cheklovlarni halol hujjatlashtirish.
**Qayerda:** `notebooks/10_final_evaluation.ipynb` → `docs/evaluation_report.md`, `docs/limitations.md`.

### Savollar
**Clustering:** cluster'lar ichida izchilmi? Bir-biridan farq qiladimi? `k` o'zgarsa nima bo'ladi? Representative chunk'lar rostdan vakilmi?
**Representation:** TF-IDF vs Embeddings — qaysi biri qaysi turdagi hujjatda yaxshi?

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M17.T1** | Test to'plami (kamida 5 ta turli PDF) | Data | M15 | `data/raw/`, `data/raw2/` |
| **M17.T2** | Notebook: har PDF × har representation (tfidf/embedding) | Evaluation | M17.T1 | `notebooks/10_final_evaluation.ipynb` |
| **M17.T3** | **Translation impact**: asl (multilingual) vs tarjima (EN) | Evaluation | M-T, M17.T2 | `notebooks/10_final_evaluation.ipynb` |
| **M17.T4** | Manual inspection rubric (K1/K2/K3, 1–5, har cluster) | Evaluation | M17.T2 | `notebooks/10_final_evaluation.ipynb` |
| **M17.T5** | Stability (5 seed) + LLM hallucination tekshiruvi (10 misol) | Evaluation | M17.T4 | `notebooks/10_final_evaluation.ipynb` |
| **M17.T6** | Performance jadvali (vaqt/xotira) | Evaluation | M17.T5 | `notebooks/10_final_evaluation.ipynb` |
| **M17.T7** | `evaluation_report.md` + `limitations.md` | Docs | M17.T6 | `docs/evaluation_report.md`, `docs/limitations.md` |

**Muhim:** Unsupervised'da "to'g'ri javob" yo'q — **miqdoriy + sifat** baholash birga. `k` va translation rejimi **sabab bilan** hujjatlashtiriladi.
### Qadamlar
- [ ] **Test to'plami:** kamida 5 ta turli PDF (ilmiy maqola, texnik hujjat, hisobot, o'quv material, kitob bobi).
- [ ] Har PDF × har representation (TF-IDF / embedding) uchun: silhouette, inertia, o'lcham taqsimoti.
- [ ] **Manual inspection rubric** (har cluster uchun 1–5 baho):
  - Izchillik: chunk'lar bir mavzumi?
  - Aniqlik: top terms mavzuni aks ettiradimi?
  - Farqlanish: boshqa cluster'dan ajralib turadimi?
- [ ] **Barqarorlik:** 5 xil `random_state` — cluster'lar qanchalik o'zgaradi? (ixtiyoriy: Adjusted Rand Index)
- [ ] **LLM sifati:** label/summary cluster mazmuniga mos-mi? "Gallyutsinatsiya" (matnda yo'q narsa) bormi? 10 ta misolni qo'lda tekshir.
- [ ] **Performance:** vaqt/xotira (sahifa soni bo'yicha) — jadval.
- [ ] `docs/evaluation_report.md` yoz: metodika, jadvallar, grafiklar, xulosalar.
- [ ] `docs/limitations.md` yoz.

### Muhim cheklov (hisobotga kiritilsin)
Unsupervised learning'da odatda "to'g'ri javob" yo'q — shuning uchun **miqdoriy + sifat** baholash birga ishlatiladi.

### Definition of Done
Hujjatlashtirilgan baholash va cheklovlar mavjud.

**Commit:** `M17: final ML evaluation report and limitations`

---

## M18 — Production cleanup

**Maqsad:** prototipni toza muhandislik loyihasiga aylantirish.
**Qayerda:** butun loyiha.

### Task breakdown

| Task | Nomi | Tur | Deps | Fayllar |
|------|------|-----|------|--------|
| **M18.T1** | Takrorlanuvchi kodni refaktor + type hints | Cleanup | M17 | `app/**` |
| **M18.T2** | `logging.py` — `print` → logger | Cleanup | M18.T1 | `app/core/logging.py` |
| **M18.T3** | Config to'liq `pydantic-settings` da, hardcode yo'q | Cleanup | M18.T2 | `app/core/config.py` |
| **M18.T4** | Xatolarni qayta ko'rish (ichki tafsilot chiqmasin) | Cleanup | M18.T3 | `app/**` |
| **M18.T5** | `/docs` misollari, API sxemalari | Polish | M18.T4 | `app/schemas/analysis.py` |
| **M18.T6** | `architecture.md` yakuniy (diagrammalar + §52 sarlavhalari) | Docs | M18.T5 | `docs/architecture.md` |
| **M18.T7** | **`API.md`** — endpoint kontraktlari (OpenAPI asosida) | Docs | M18.T6 | `API.md` |
| **M18.T8** | **`CONTRIBUTING.md`** — hisob-kitob, workflow, qoidalar | Docs | M18.T7 | `CONTRIBUTING.md` |
| **M18.T9** | README final (o'rnatish/run/Docker/test/natija) | Docs | M18.T8 | `README.md` |
| **M18.T10** | "Yangi odam 10 daqiqada ishga tushira oladimi?" | Verify | M18.T9 | — |

**Majburiy hujjat to'plami (`AGENTS.md` §33):**

```text
README.md            ← ushbu fayl (M0 dan)
docs/architecture.md ← M0 da yozilgan, M18 da yangilanadi
docs/io_contract.md  ← M0 da yozilgan, M18 da yangilanadi
docs/architecture/ADR-00N-*.md ← M0 da 5 ta, M18 da qo'shiladi
API.md               ← M18.T7
CONTRIBUTING.md      ← M18.T8
```

**Muhim:** `docs/architecture.md` va `docs/io_contract.md` allaqachon yozilgan (M0) — M18 da **yangilanadi**, noldan yozilmaydi.
> `AGENTS.md` §33: *"Documentation MUST describe reality, not intended future behavior."* — M18 da hamma narsa **ishlayotgan** holatda tasdiqlanadi.
### Qadamlar
- [ ] Takrorlanuvchi kodni refaktor qil.
- [ ] Hamma funksiyalarga **type hints** (`mypy` bilan tekshirish — ixtiyoriy).
- [ ] **Logging** (`app/core/logging.py`): `print` ni logger bilan almashtir, har bosqich vaqti.
- [ ] **Konfiguratsiya:** hamma sozlama `pydantic-settings` orqali (`app/core/config.py`), hardcode yo'q.
- [ ] Xatolarni qayta ko'rib chiq, ichki tafsilot foydalanuvchiga chiqmasin.
- [ ] API sxemalari va `/docs` tavsiflarini yaxshila (example'lar).
- [ ] `examples/` yoki `data/raw/` da namuna PDF'lar (litsenziyasi ruxsat bergan).
- [ ] **Arxitektura diagrammasi** (`docs/architecture.md`).
- [ ] **Cheklovlar** va **performance** hujjati.
- [ ] README: o'rnatish, ishga tushirish, Docker, test, namuna natija skrinshoti.
- [ ] Linter/formatter (`ruff`, `black`) — ixtiyoriy.
- [ ] Yangi odam `git clone` qilib 10 daqiqada ishga tushira oladimi? Tekshir (yoki do'stingga ber).

### Definition of Done
Boshqa dasturchi clone qilib, sozlab, ishga tushira oladi va loyihani tushunadi.

**Commit:** `M18: production cleanup and documentation`

---

# 8. Baholash protokoli

Har cluster-natijasini quyidagi tartibda tekshir:

```text
1. Cluster o'lchamlari     → bitta cluster >70% yoki <3 chunk'li cluster bormi?
2. Silhouette (cosine)     → >0.05 ma'noli signal; <0 yomon belgi (matn uchun mutlaq qiymatlar past bo'ladi)
3. Elbow                   → egilish nuqtasi bormi?
4. Stability               → 5 xil seed → o'xshash cluster'larmi?
5. Top terms               → mavzuni ifodalaydimi?
6. Representative chunks   → qo'lda o'qi: bir mavzumi?
7. Qaror                   → raqam + kuzatuv bilan asosla, docs'ga yoz
```

**Qizil bayroqlar:**
- Top terms'da `the`, `of`, `figure`, `page` → preprocessing/`max_df` ga qayt.
- Cluster'lar sahifa tartibi bilan mos tushyapti (har cluster = ketma-ket sahifalar) → hujjat strukturasi, mavzu emas; chunk/representation'ni qayta ko'r.
- Hamma chunk bitta cluster'da → `k` kichik yoki vektorlash yomon.

---

# 9. Edge case'lar

> Barcha holat **M15** da test qilinadi.

| Holat | Kutilgan xulq | Qayerda ushlanadi |
|-------|---------------|-------------------|
| Bo'sh PDF | `ScannedPdfError` / 422 | M2 |
| Skanerlangan PDF (rasm) | aniq xabar, OCR yo'q | M2 |
| Juda qisqa PDF | `NotEnoughContentError` | M4/M12 |
| Juda katta PDF | hajm/sahifa chegarasi, 413 | M13 |
| Yaroqsiz fayl | `InvalidPdfError` / 400 | M2/M13 |
| Kam chunk (< 10) | clustering'ni o'tkazib yuborish yoki xato | M12 |
| `k` ≥ chunk soni | `k` ni avtomatik kamaytirish | M7/M12 |
| Ingliz bo'lmagan matn | `stop_words=None` + ko'p tilli embedding | M5/M9/M-T |
| **Aralash tilli PDF** | Default: ko'p tilli embedding. Manual inspection yomon bo'lsa → `TRANSLATION_ENABLED=true` (page/document) | M-T/M12/M17 |
| **Tarjima provider ishlamadi** | `TranslationUnavailableError` → **503**. Default `Noop` — xavfsiz (tashqi so'rov yo'q) | M-T/M13 |
| **Tarjima — chunk level (noto'g'ri)** | **TAQIQLANGAN.** Faqat `document` \| `page`. Boshqa qiymat → `TranslationError` | M-T |
| Parol bilan himoyalangan | `InvalidPdfError` | M2 |
| Ollama o'chiq | qisman natija (ML) + 503/ogohlantirish | M11–M13 |

---

# 10. Git qoidalari

- Bitta milestone = bir nechta kichik commit (`M5: ...` prefiksi bilan).
- Har milestone oxirida **tag**: `git tag m5-done`.
- Katta PDF va `data/processed/` ni commit qilma.
- `main` har doim ishlaydigan holatda; yangi narsalar uchun `feature/mN-nom` branch (ixtiyoriy).

---

# 11. Non-goals

Birinchi versiyada **qurilmaydi**:
- Authentication, to'lovlar, ko'p foydalanuvchili akkauntlar
- Redis, Kubernetes, microservice'lar, distributed processing
- Custom LLM training / fine-tuning
- OCR (skanerlangan PDF)

Birinchi maqsad: **to'g'ri, tushunarli va o'lchanadigan ML pipeline.**

---

# 12. Kelajak

```text
MVP → Yaxshiroq chunking → Yaxshiroq embeddings → DBSCAN/HDBSCAN
    → LDA/NMF → RAG → Hujjatlarni solishtirish → Ko'p hujjatli tahlil
    → Savol-javob → Manbali (citation) javoblar → Evaluation dashboard
```

**Keyin o'rganiladigan algoritmlar:** DBSCAN, HDBSCAN, Agglomerative, LDA, NMF, hybrid representations, RAG, ML observability.

### O'rganish kurikulumi (referens)
- **Boshlang'ich:** ML nima · Supervised vs Unsupervised · Clustering · Feature · Vector · Text preprocessing · Bag of Words · TF-IDF
- **Elementar:** K-Means · Centroid · Distance · Inertia · Silhouette · PCA · Cosine similarity · Embeddings
- **O'rta:** Topic discovery · Semantic clustering · Cluster evaluation · NLP pipelines · Chunking · Similarity search · LLM integratsiya · Prompt engineering
- **Ilg'or:** DBSCAN · HDBSCAN · Agglomerative · LDA · NMF · Hybrid · RAG · Evaluation framework'lar · Production ML · Observability

---

# 13. Progress tracker

> **Har sessiya boshlashda** shu tracker'dan birinchi belgilanmagan `[ ]` taskni top va shu task ustida ishlashni boshla.
> Milestone tugagach `⬜` ni `✅` ga almashtir va sanani yoz.

### Umumiy holat

| Holat | Milestone | Sana |
|-------|-----------|------|
| ✅ | **M0** — Planning | 2026-10-04 |
| ✅ | **M1** — Environment | 2026-10-04 |
| ✅ | **M-T** — Translation Module | 2026-10-04 |
| ✅ | **M2** — PDF Extraction | 2026-10-04 |
| ✅ | **M3** — Text Preprocessing | 2026-10-04 |
| ⬜ | **M4** — Chunking | ____ |
| ⬜ | **M5** — TF-IDF | ____ |
| ⬜ | **M6** — K-Means | ____ |
| ⬜ | **M7** — Clustering Evaluation | ____ |
| ⬜ | **M8** — PCA/SVD Vizualizatsiya | ____ |
| ⬜ | **M9** — Embeddings | ____ |
| ⬜ | **M10** — Topic Discovery | ____ |
| ⬜ | **M11** — Ollama | ____ |
| ⬜ | **M12** — Summary Pipeline | ____ |
| ⬜ | **M13** — FastAPI | ____ |
| ⬜ | **M14** — Streamlit | ____ |
| ⬜ | **M15** — Testing | ____ |
| ⬜ | **M16** — Docker | ____ |
| ⬜ | **M17** — Final ML Evaluation | ____ |
| ⬜ | **M18** — Production Cleanup | ____ |

### Task darajasida holat

```text
M0   ✅ T1 Scope/Goals/Non-goals + arxitektura brief   → docs/architecture.md
     ✅ T2 Input/Output contract + baholash mezonlari  → docs/io_contract.md
     ✅ T3 Namuna PDF'lar (4 ta, aralash tilli bilan)   → data/raw/
     ✅ T4 ADR'lar (5 ta arxitektura qarori)            → docs/architecture/

M1   ✅ T1 venv (Python 3.12) + requirements.txt        → requirements.txt
     ✅ T2 Papka tuzilmasi (+ app/translation/)         → app/
     ✅ T3 .gitignore + .env.example                    → .env.example
     ✅ T4 Muhit + Ollama tekshiruvi                    → qwen3:4b ishlayapti

M-T  ✅ T1 Translation exceptions + base interface   → app/translation/{exceptions,base}.py
     ✅ T2 NoopTranslator (default)                   → app/translation/noop.py
     ✅ T3 GoogleTranslateTranslator                  → app/translation/google_translate.py
     ✅ T4 Factory + public API                       → app/translation/factory.py
     ✅ T5 Config integratsiyasi                      → app/core/config.py, .env.example
     ✅ T6 Unit testlar (53) + integration (1)        → tests/test_translation.py, pytest.ini
     ✅ T7 Notebook: original vs translated            → notebooks/11_translation_experiments.ipynb

M2   ✅ T1 Notebook 01 (PyMuPDF)                    → notebooks/01_pdf_extraction.ipynb
     ✅ T2 PageText sxemasi + exceptions            → app/schemas/document.py, app/core/exceptions.py
     ✅ T3 extract_pages() + magic/needs_pass        → app/services/pdf_extractor.py, app/core/config.py
     ✅ T4 Unit testlar (45), pdf_extractor 100%     → tests/test_pdf_extractor.py

M3   ✅ T1 Notebook 02 (artefaktlar + 3 strategiya)  → notebooks/02_preprocessing.ipynb
     ✅ T2 preprocessing.py (7 funksiya)              → app/nlp/preprocessing.py
     ✅ T3 display_text / ml_text ajratish            → app/nlp/preprocessing.py
     ✅ T4 Unit testlar (68), app/nlp 100%           → tests/test_preprocessing.py

M4   ⬜ T1 Notebook 03 (parametr solishtirish)
     ⬜ T2 Chunk sxemasi + chunking.py
     ⬜ T3 config.py ga parametrlar
     ⬜ T4 Unit testlar

M5   ⬜ T1 Notebook 04 (qo'lda → NumPy → sklearn)
     ⬜ T2 build_tfidf()
     ⬜ T3 stop_words tilga bog'liq
     ⬜ T4 Unit testlar

M6   ⬜ T1 Notebook 05 (2D qo'lda → NumPy → sklearn)
     ⬜ T2 cluster_kmeans()
     ⬜ T3 top_terms()
     ⬜ T4 representative_chunks()
     ⬜ T5 Unit testlar

M7   ⬜ T1 Notebook 06 (elbow + silhouette + stability)
     ⬜ T2 evaluate_k_range()
     ⬜ T3 suggest_k()
     ⬜ T4 Qarorni yozish

M8   ⬜ T1 Notebook 07 (SVD/PCA)
     ⬜ T2 plot_clusters_2d()

M9   ⬜ T1 Notebook 08 (embedding + cosine)
     ⬜ T2 build_embeddings()
     ⬜ T3 Ko'p tilli rejim
     ⬜ T4 TF-IDF vs Embedding jadvali
     ⬜ T5 Unit testlar

M10  ⬜ T1 Notebook 09 (main topic qoidasi)
     ⬜ T2 Topic sxemasi
     ⬜ T3 discover_topics()
     ⬜ T4 Embedding uchun top terms
     ⬜ T5 Representative chunk'lar
     ⬜ T6 Unit testlar

M11  ⬜ T1 Notebook (Ollama API)
     ⬜ T2 Exceptions + base interface
     ⬜ T3 OllamaClient (timeout/retry)
     ⬜ T4 prompts.py (3 prompt)
     ⬜ T5 generate_json() + validatsiya
     ⬜ T6 Unit + integration testlar

M12  ⬜ T1 AnalysisResult sxemasi
     ⬜ T2 analysis_service.py
     ⬜ T3 Translation integratsiyasi
     ⬜ T4 Avtomatik k + NotEnoughContentError
     ⬜ T5 Timing + logging
     ⬜ T6 Graceful degradation
     ⬜ T7 E2E testlar

M13  ⬜ T1 FastAPI app + lifespan
     ⬜ T2 Fayl validatsiyasi (magic bytes, filename sanitize)
     ⬜ T3 POST /documents/analyze
     ⬜ T4 GET /health
     ⬜ T5 Xato xaritasi (+ Translation → 503)
     ⬜ T6 CORS aniq + rate limit
     ⬜ T7 API testlar

M14  ⬜ T1 api_client.py (timeout + xato normalizatsiyasi)
     ⬜ T2 UI skeleti
     ⬜ T3 Natija ko'rsatish
     ⬜ T4 Xatolarni ko'rsatish
     ⬜ T5 Grafiklar + session_state

M15  ⬜ T1 Fixture PDF'lar
     ⬜ T2 pytest.ini + conftest.py
     ⬜ T3 ML property testlari
     ⬜ T4 Edge case'lar
     ⬜ T5 E2E + translation yoqiq/yoqilgan
     ⬜ T6 Coverage ~70%+

M16  ⬜ T1 .dockerignore
     ⬜ T2 Backend Dockerfile (3.12, CPU torch, non-root)
     ⬜ T3 Frontend image
     ⬜ T4 docker-compose.yml
     ⬜ T5 Ollama wiring + HF cache
     ⬜ T6 Resource limits + healthcheck
     ⬜ T7 Port exposure policy (Ollama ichki)
     ⬜ T8 Toza muhit testi + health tekshiruvi

M17  ⬜ T1 Test to'plami (5 ta PDF)
     ⬜ T2 Notebook 10 (har PDF × har representation)
     ⬜ T3 Translation impact solishtirish
     ⬜ T4 Manual rubric (K1/K2/K3)
     ⬜ T5 Stability + LLM hallucination
     ⬜ T6 Performance jadvali
     ⬜ T7 evaluation_report.md + limitations.md

M18  ⬜ T1 Refaktor + type hints
     ⬜ T2 logging.py
     ⬜ T3 Config to'liq
     ⬜ T4 Xatolarni qayta ko'rish
     ⬜ T5 /docs misollari
     ⬜ T6 architecture.md yakuniy (§52 sarlavhalari)
     ⬜ T7 API.md (OpenAPI asosida)
     ⬜ T8 CONTRIBUTING.md
     ⬜ T9 README final
     ⬜ T10 "10 daqiqada ishga tushirish" testi
```

**Hozirgi qadam: M4.T1** — Notebook 03 (chunk parametrlari: 200/800, 300/1200, 500/2000).

> M0, M1, M-T, M2, M3 ✅. M3 dan keyin pipeline matnni **to'g'ri
> tozalaydi** va TF-IDF uchun tayyor matn beradi.
> `PDF → sahifa matni → toza matn`. Keyingi: chunk → TF-IDF → K-Means.

- [ ] PDF upload ishlaydi
- [ ] Matn ajratish ishlaydi
- [x] Preprocessing ishlaydi
- [ ] Chunking ishlaydi
- [ ] TF-IDF ishlaydi
- [ ] K-Means ishlaydi
- [ ] Cluster sifati o'lchanadi
- [ ] PCA vizualizatsiya ishlaydi
- [ ] Embeddings ishlaydi
- [ ] Topic discovery ishlaydi
- [ ] Ko'p tilli PDF ishlaydi (multilingual pipeline)
- [ ] Translation moduli ishlaydi (pluggable, default ochiq)
- [ ] Ollama integratsiyasi ishlaydi
- [ ] Summary generatsiyasi ishlaydi
- [ ] FastAPI ishlaydi
- [ ] Streamlit ishlaydi
- [ ] Testlar o'tadi
- [ ] Docker ishlaydi
- [ ] ML baholash hujjatlashtirilgan
- [ ] Cheklovlar hujjatlashtirilgan
- [ ] README to'liq

---

# 14. Ish jurnali

Har sessiyadan keyin shu shablonni ko'chirib to'ldir (`docs/learning_notes.md` ga):

```markdown
## 2026-__-__ · M_ · ___ soat
- Nima qildim:
- Nimani tushunmadim / qiyin bo'ldi:
- Natija (raqam/kuzatuv):
- Ertaga birinchi qadam:
```

---

# 15. Lug'at

| Atama | Qisqa izoh |
|-------|------------|
| Unsupervised learning | label'siz ma'lumotdan tuzilma topish |
| Chunk | hujjatning tahlil birligi (paragraf/guruh) |
| Vocabulary | korpusdagi barcha noyob so'zlar |
| TF-IDF | so'zning hujjatdagi muhimligi (ko'p uchraydi, lekin hamma joyda emas) |
| Sparse matritsa | asosan nollardan iborat, faqat nolmas qiymatlar saqlanadi |
| Centroid | cluster markazi (o'rtacha vektor) |
| Inertia | cluster ichidagi kvadrat masofalar yig'indisi |
| Silhouette | nuqta o'z cluster'iga qanchalik mos ekanini o'lchaydi (−1…1) |
| Cosine similarity | ikki vektor orasidagi burchak kosinusi (yo'nalish o'xshashligi) |
| Embedding | matn ma'nosini ifodalovchi zich vektor |
| PCA / SVD | o'lchamni kamaytirish usuli |
| Representative chunk | cluster markaziga eng yaqin chunk |
| Topic | cluster'ning odam tushunadigan mavzu talqini |
| Graceful degradation | bir qism yiqilganda ham qisman natija qaytarish |

---

> **Esdan chiqarma:** maqsad — ishlaydigan ilova ham, lekin undan muhimi — **har bir komponent nega bor, nima qiladi, qanday ishlaydi va qanday baholanishini tushunib** tugatish.