# Architecture Brief — PDF Analyzer (Unsupervised Learning)

**Version:** 1.0.0
**Status:** M0 — tasdiqlangan
**Scope:** Butun loyiha (M0–M18)
**Manba:** `README.md` (ish daftari), `opencode/AGENTS.md` (Global Engineering Constitution)

---

# 1. Problem

Uzun PDF hujjatlarda mavzularni **oldindan belgilangan label'larsiz** (unsupervised) aniqlash kerak.

An'anaviy yondashuvda mavzularni LLM beradi (`PDF → LLM → "mana bir nechta mavzular"`). Bu yondashuv:

- **Takrorlanmaydi** — har so‘rovda boshqa mavzular chiqadi.
- **Tekshirilmaydi** — natijani miqdoriy o‘lchash mumkin emas.
- **Halol emas** — LLM matnda yo‘q narsani "topishi" mumkin (gallyutsinatsiya).
- **Interaktiv emas** — foydalanuvchi "qanday tuzilma bor?" degan savolga javob olmaydi.

**Hal qilinadigan muammo:** Foydalanuvchiga PDF ichidagi **haqiqiy tuzilmani** (mavzular, ularning o‘lchami, vakili, joylashuvi) ko‘rsatish, shu tuzilmani LLM orqali **tushuntirish** — lekin tuzilmani **ML topishi**, LLM emas.

---

# 2. Goals

| # | Maqsad |
|---|--------|
| G1 | **Unsupervised-first** — mavzular ML (clustering) topadi, LLM faqat tushuntiradi. |
| G2 | **Tushunarli natija** — har cluster uchun top terms, representative chunk’lar, sahifa raqamlari. |
| G3 | **O‘lchanadigan sifat** — quantitative (silhouette, inertia, stability) + qualitative (manual inspection) birga. |
| G4 | **Reproducible** — bir xil kirish → bir xil natija (`random_state=42`). |
| G5 | **Modullar ajratilgan** — PDF / Translation / NLP / ML / LLM / Service / API qatlamlari bir-birini bilmaydi. |
| G6 | **Local-first** — default rejimda tashqi servis (internet) kerak emas. |
| G7 | **Pluggable** — Translation va LLM provider’lari config orqali almashtiriladi. |
| G8 | **Graceful degradation** — LLM/translation ishlamasa ham ML natijasi saqlanadi. |
| G9 | **O‘rganish maqsadli** — har komponent 10-qadamli sikl bilan qo‘lda tushuniladi (manual-first). |
| G10 | **Ishlaydigan natija** — har milestone oxirida demo mumkin (incremental). |

---

# 3. Non-Goals

Birinchi versiyada **qilinmaydigan** narsalar:

| Non-Goal | Sababi |
|-----------|--------|
| OCR (Tesseract/PaddleOCR) | Skanerlangan PDF qo‘shimcha subsystema; MVP da faqat matnli PDF. |
| Multi-PDF batch tahlil | Bitta fayl = bitta tahlil. |
| PDF saqlash, DB, foydalanuvchi auth | Stateless prototip; natija qaytariladi, saqlanmaydi. |
| Topic modeling (LDA, BERTopic) | Asosiy yondashuv — K-Means + explanation. |
| Streaming LLM output | `stream=False` + structured JSON yetarli. |
| Chat / Q&A (RAG) | Statik tahlil (topics, summary, key points). |
| LangChain / LangGraph | Minimal dependencies — to‘g‘ridan-to‘g‘ri `httpx` + FastAPI. |
| NER / Relation Extraction | Scope tashqarisi. |
| Supervised classification | Unsupervised tamoyiliga zid. |
| Redis / Kubernetes / microservices | Yakka server yetarli. |
| To‘liq offline LLM majburiyligi | Ollama default, lekin cloud provider ham qo‘llab-quvvatlanadi. |

---

# 4. Requirements

## 4.1 Functional Requirements

| ID | Talab | Milestone |
|----|-------|-----------|
| FR1 | PDF fayldan sahifa-darajadagi matn olish (PyMuPDF). | M2 |
| FR2 | Skanerlangan/bo‘sh/buzilgan PDF uchun **aniq** xato berish. | M2 |
| FR3 | Matnni tozalash: unicode, hyphenation, header/footer, sahifa raqamlari. | M3 |
| FR4 | Ikki matn turi saqlash: `display_text` (yengil) va `ml_text` (jiddiy). | M3 |
| FR5 | Paragraf-asosida chunk’lash, har chunk’ga sahifa raqamlari biriktirish. | M4 |
| FR6 | Matnni vektorga aylantirish: TF-IDF **va** semantik embedding. | M5, M9 |
| FR7 | K-Means bilan cluster’lash; top terms + representative chunk’lar olish. | M6 |
| FR8 | `k` ni asoslab tanlash (elbow + silhouette + stability + qo‘lda ko‘rish). | M7 |
| FR9 | 2D vizualizatsiya (SVD/PCA) — izohlangan. | M8 |
| FR10 | Anonymous cluster’larni **Topic** ob’ektlariga aylantirish. | M10 |
| FR11 | LLM orqali topic label/description, key points, summary — **structured JSON**. | M11 |
| FR12 | Bitta funksiya bilan end-to-end tahlil. | M12 |
| FR13 | HTTP API: `POST /documents/analyze`, `GET /health`. | M13 |
| FR14 | Streamlit UI: yuklash → tahlil → natija ko‘rsatish. | M14 |
| FR15 | **Ixtiyoriy tarjima** — document/page level, pluggable provider. | M-T |
| FR16 | Turli tilli PDF’ni qo‘llab-quvvatlash (multilingual pipeline). | M-T, M9 |

## 4.2 Non-Functional Requirements

| ID | Talab |
|----|-------|
| NFR1 | **Reproducibility** — barcha tasodifiy joylarda `random_state=42`. |
| NFR2 | **Determinizm testlari** — bir xil `random_state` → bir xil label’lar. |
| NFR3 | **Qatlamlar ajratilgan** — qatlamlar bir-birining ichki tafsilotini bilmaydi. |
| NFR4 | **Konfiguratsiya** — barcha sozlamalar `pydantic-settings` orqali, hardcode yo‘q. |
| NFR5 | **Xato izolyatsiyasi** — ichki tafsilot foydalanuvchiga chiqmaydi (500). |
| NFR6 | **Tezkor feedback** — og‘ir hisob event loop’ni bloklamasdi (`def` endpoint / threadpool). |
| NFR7 | **Test qamrovi** — `app/` asosiy mantiqda ~70%+. |
| NFR8 | **Observability** — har bosqich vaqti o‘lchanadi va loglanadi. |
| NFR9 | **Local-first** — default konfiguratsiyada internet talab qilinmaydi. |
| NFR10 | **Graceful degradation** — tashqi servis xatosi → ML natijasi saqlanadi. |

---

# 5. Constraints

| # | Cheklov |
|---|---------|
| C1 | Faqat **matnli** PDF (skanerlangan qo‘llab-quvvatlanmaydi). |
| C2 | Boshlang‘ich til — ingliz; boshqa tillar uchun multilingual pipeline (M-T). |
| C3 | Python **3.10+** (lokal muhitda 3.14). |
| C4 | LLM — lokal Ollama (default), lekin cloud provider ham qo‘llab-quvvatlanadi. |
| C5 | Boshlang‘ich PDF hajmi ≤ ~100 bet (katta PDF’lar keyin). |
| C6 | Xarajatni minimallashtirish — test/tajribada free tier yetarli, keyinchalik cloud. |
| C7 | Maxfiylik — default rejimda matn tashqariga chiqmaydi. |
| C8 | Yoshlanish (ta’lim) maqsadi — kod tushunarli va qayta ishlatiladigan bo‘lishi shart. |

---

# 6. Architecture

## 6.1 Pipeline

```text
                          PDF
                           │
                           ▼
                  ┌─────────────────┐
                  │ PDF Text Extract │   M2  (PyMuPDF)
                  └────────┬────────┘
                           ▼
                ┌───────────────────────┐
                │ [OPTIONAL] Translate   │  M-T  (document/page level)
                │  noop | google_translate │
                └────────┬───────────────┘
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
                      LLM (M11)           pluggable: ollama | openai | ...
                           ▼
                  Final Explanation        M12
                           ▼
                       FastAPI            M13
                           ▼
                     Streamlit UI          M14
```

## 6.2 Yondashuv tanlovi (mavzu topish)

```text
✅ TO'G'RI:   PDF → Unsupervised ML → Topilgan tuzilma → LLM tushuntiradi
❌ NOTO'G'RI: PDF → LLM → "mana bir nechta mavzular"
```

---

# 7. Components

| Qatlam | Papka | Mas’uliyat | **Bilmasligi kerak** |
|--------|-------|------------|----------------------|
| **PDF** | `app/services/pdf_extractor.py` | PDF → sahifa matni | ML, LLM, API, Translation |
| **Translation** | `app/translation/` | Pluggable tarjima (noop / google_translate), document/page level | ML, API, pipeline ichki tafsilotlari |
| **NLP** | `app/nlp/` | Tozalash, chunk’lash | ML, LLM, API, Translation |
| **ML** | `app/ml/` | Vektorlash, clustering, baholash, vizualizatsiya, topics | PDF, LLM, API, Translation |
| **LLM** | `app/llm/` | LLM aloqa (pluggable), prompt’lar, structured JSON | ML algoritmlarining ichki tafsilotlari |
| **Service** | `app/services/` | Pipeline orchestrator | HTTP tafsilotlari |
| **API** | `app/api/` | HTTP, validatsiya, xato xaritasi | ML/Translation ichki tafsilotlari |
| **Schemas** | `app/schemas/` | Pydantic model — contracts | Barchasining ichki logikasi |

---

# 8. Data Flow

```text
[bytes/str PDF]
      │
      ▼ extract_pages()          M2
[list[PageText(page, text)]]
      │
      ▼ translate()  (ixtiyoriy) M-T  ← document yoki page level
[list[PageText]]  (translated yoki asl)
      │
      ▼ clean_text() / ml_text()  M3
[display_text, ml_text]
      │
      ▼ build_chunks()            M4
[list[Chunk(id, text, pages, n_chars)]
      │
      ├──▶ build_tfidf(ml_text)   M5   → sparse matrix X_tfidf
      └──▶ build_embeddings(display_text) M9 → dense matrix X_emb
      │
      ▼ cluster_kmeans(X, k)      M6
[labels]
      │
      ▼ evaluate_k_range()        M7  (k=None bo'lsa avtomatik)
[k, inertia, silhouette]
      │
      ▼ discover_topics()         M10
[list[Topic(cluster_id, size, share, top_terms,
           representative_chunks, pages, label=None, description=None)]
      │
      ▼ LLM: label / key_points / summary    M11
[Topic.label, Topic.description, key_points, main_topic, summary]
      ▼
[AnalysisResult]                 M12
      │
      ▼ FastAPI response           M13
      ▼ Streamlit UI               M14
```

---

# 9. API Boundaries

## 9.1 Endpoint’lar

| Method | Path | Vazifa | Kod |
|--------|------|--------|-----|
| POST | `/documents/analyze` | PDF → to‘liq tahlil | 200 / 400 / 413 / 422 / 503 / 500 |
| GET | `/health` | Tiriklik + Ollama holati | 200 / 503 |
| POST | `/documents/upload` | Keyingi bosqich (saqlash) | — |
| GET | `/documents/{id}` | Keyingi bosqich (natijani olish) | — |

## 9.2 Xato xaritasi

| Ichki xato | HTTP | Sababi |
|------------|------|--------|
| `InvalidPdfError` | **400** | Fayl PDF emas yoki buzilgan |
| (hajm chegarasi) | **413** | `MAX_UPLOAD_MB` dan oshgan |
| `ScannedPdfError` | **422** | Skanerlangan (matn qatlami yo‘q) |
| `NotEnoughContentError` | **422** | Chunk soni < 10, clustering ma’nosiz |
| `OllamaUnavailableError` | **503** | LLM mavjud emas (ML natijasi saqlanadi) |
| `TranslationError` / `TranslationUnavailableError` | **503** | Tashqi translation provider ishlamadi |
| Kutilmagan xato | **500** | Logga yoziladi, foydalanuvchiga ichki tafsilot **chiqmaydi** |

## 9.3 Qatlam chegaralari

```text
API qatlami  → faqat HTTP va validatsiya bilan shug‘ullanadi
Service      → faqat pipeline tartibini boshqaradi
ML / NLP     → HTTP haqida **bilmaydi**
LLM / Trans  → ML ichki tafsilotlarini **bilmaydi**
```

---

# 10. Data Storage

MVP da **stateless** — hech narsa saqlanmaydi. Bu atama arxitekturaning eng oddiy va eng xavfsiz tanlovi.

| Ma’lumot | Joy | Muddat | Sabab |
|----------|-----|--------|-------|
| Namuna PDF’lar | `data/raw/` | Doimiy (lokal) | Test uchun |
| Keshlangan chunk/vektorlar | `data/processed/` | Vaqtinchalik | `.gitignore`’da |
| Test fixture PDF’lar | `tests/fixtures/` | Doimiy (kichik, <200KB) | Git’da saqlanadi |
| API natijalari | — | **Saqlanmaydi** | Stateless; foydalanuvchi oladi |
| Model keshi (HF) | `~/.cache/huggingface` | Doimiy | Docker’da volume |

> Kelajakda (Non-Goals): SQLite/Postgres, foydalanuvchi tarixi, natija arxivi.

---

# 11. Failure Modes

| # | Failure mode | Kechilish chorasi | Qayerda |
|---|--------------|--------------------|---------|
| F1 | **Skanerlangan PDF** (matn yo‘q) | Jami belgi < `MIN_TOTAL_CHARS` → `ScannedPdfError` | M2 |
| F2 | **Buzilgan/yaroqsiz PDF** | PyMuPDF xatosi → `InvalidPdfError` | M2 |
| F3 | **Parol bilan himoyalangan PDF** | `InvalidPdfError` | M2 |
| F4 | **Juda kam chunk** (< 10) | `NotEnoughContentError` (clustering ma’nosiz) | M4/M12 |
| F5 | **`k` > chunk soni** | `k_max = min(10, n_chunks // 5)` bilan cheklash | M7/M12 |
| F6 | **Bitta cluster 80%+** | Notekis taqsimot — signal; qayta baholash | M6/M7 |
| F7 | **Top terms — `the, of, page`** | `max_df`/`stop_words`/`min_df` ga qaytish | M5 |
| F8 | **Cluster = ketma-ket sahifalar** | Bu mavzu emas, hujjat strukturasi — chunk/representation qayta ko‘rish | M7 |
| F9 | **Silhouette juda past** | Matn uchun tabiiy; trend va qo‘lda ko‘rish bilan baholash | M7 |
| F10 | **LLM o‘chiq / timeout** | **Graceful degradation**: ML natijasi qaytariladi, LLM maydonlari `None` | M11/M12 |
| F11 | **LLM noto‘g‘ri JSON** | Retry (1–2 marta) + Pydantic validation → keyin graceful degrade | M11 |
| F12 | **Translation provider ishlamadi** | `TranslationUnavailableError` → 503; default `Noop` xavfsiz | M-T/M13 |
| F13 | **Ikki ustunli PDF, matn tartibi buzilgan** | Cheklov sifatida hujjatlashtirish (`docs/limitations.md`) | M2/M17 |
| F14 | **Uzun PDF, ko‘p vaqt** | Sahifa chegarasi + timing log + progressor | M12/M13 |
| F15 | **Embedding model yuklanmadi** (offlinda) | Aniq xato, `.env` orqali model yo‘lini tekshirish | M9/M13 |
| F16 | **Event loop bloklandi** | `def` endpoint (threadpool) yoki `run_in_threadpool` | M13 |

---

# 12. Security

| # | choralar |
|---|-----------|
| S1 | **Secrets faqat `.env`** — `.env` `.gitignore`’da, hech qachon commit qilinmaydi. |
| S2 | **API key’lar loglanmaydi** va xatolarda chiqmaydi. |
| S3 | **Ichki tafsilot foydalanuvchiga chiqmaydi** — 500 da faqat umumiy xabar; tafsilot logda. |
| S4 | **Fayl validatsiyasi** — kengaytma, `content_type`, hajm, **magic bytes `%PDF`**. |
| S5 | **Maxfiylik default’da** — translation o‘chirilgan, LLM lokal; matn tashqariga chiqmaydi. |
| S6 | **O‘rnatilgan ilgoritmlar** — `pdf` faylga kod bajarilmaydi, faqat parse qilinadi. |
| S7 | **Prompt injection chegarasi** — LLM’ga faqat qisqa, belgilangan kontekst beriladi; “faqat berilgan matnga tayan” talabi. |
| S8 | **Dependency minimal** — LangChain kabi keng kutubxonalar ishlatilmaydi (attack surface kichik). |
| S9 | **Docker’da foydalanuvchi root emas**, `localhost` konteyner ichida o‘zini bildiradi. |

---

# 13. Scalability

MVP bitta server uchun mo‘ljallangan. Masshtablash yo‘llari:

| Yo‘l | Qachon | Qanday |
|-----|--------|--------|
| **Model keshi** | Hozir | Embedding model bir marta yuklanadi (`_model` global cache + FastAPI `lifespan`). |
| **Vaqtinchalik kesh** | Hozir | `data/processed/*.npy` — qayta hisoblanmaydi. |
| **Vektorlash parallel** | Kerak bo‘lganda | TF-IDF va embedding **parallel** hisoblanishi mumkin. |
| **Async navbat** | Ko‘p foydaluvchida | Hozir `def` endpoint (threadpool); keyin `asyncio.Semaphore` bilan navbat. |
| **Job queue** | Juda katta yukda | Celery/RQ + Redis (Non-Goal hozir). |
| **Model kichiklash** | Xotira muammosi | `all-MiniLM-L6-v2` (22M param) — allaqachon kichik. |
| **Chunk parallel** | Katta PDF’da | `multiprocessing` bilan chunk’lash. |
| **Stateless services** | Ko‘p instance | Natija saqlanmagani → o‘lchash oson. |

> **Qoida:** o‘lchash **kerak bo‘lganda** (o‘lchov bilan) — oldindan murakkamlash emas.

---

# 14. Observability

| # | Nima kuzatiladi | Vosita |
|---|----------------|--------|
| O1 | Har pipeline bosqichi vaqti | `time.perf_counter()` + logger |
| O2 | Qaysi bosqich sekin | Bosqich bo‘yicha timing jadvali |
| O3 | Xatolar (klass + kontekst) | `app/core/logging.py` (M18) |
| O4 | LLM/translation so‘rovlari (vaqt, natija) | Logger (javob matni emas!) |
| O5 | Klaster statistikasi | `docs/evaluation_report.md` (M17) |
| O6 | Xavfsizlik hodisalari | 400/413/500 so‘rovlar loglanadi |
| O7 | Health holati | `GET /health` (Ollama + embedding model) |

**Minimal logging formati:**

```text
LEVEL | stage | duration_ms | key=value ...
INFO  | embed | 1840        | chunks=87 model=all-MiniLM-L6-v2
ERROR | llm   | 120         | error=OllamaUnavailableError degrade=true
```

---

# 15. Deployment

| # | Bosqich | Maqsad |
|---|---------|--------|
| D1 | Lokal muhit (venv) | Development va o‘rganish |
| D2 | Lokal server (`uvicorn` + `streamlit`) | Manual test |
| D3 | Docker Compose | Toza muhitda takrorlanish (M16) |
| D4 | Public deploy | **Keyinchalik** — infra tanlash M16 dan keyin |

**Compose topologiya (M16):**

```text
docker-compose
├── backend   FastAPI + ML  (port 8000)
├── frontend  Streamlit     (port 8501)
└── ollama    (yoki host’dagi Ollama)
```

Muhim nuqtalar:
- Ollama **host**’da → `host.docker.internal:11434` (Linux: `extra_hosts`).
- Ollama **konteynerda** → `ollama` servis + `ollama_data:/root/.ollama` volume.
- **HF kesh** volume’ga olinadi (`~/.cache/huggingface`) — aks holda har build’da qayta yuklanadi.
- CPU-only torch — image hajmini kamaytiradi.

---

# 16. Risks

| # | Risk | Ehtimollik | Ta’sir | Yechim |
|---|------|-----------|--------|--------|
| R1 | **Silhouette matnda doim past** | Yuqori | O‘lchash ishonchliligini yo‘qotish | Trend + qo‘lda ko‘rish; past qiymatni “yomon” deb yorlamaslik |
| R2 | **Cluster’lar til bo‘yicha bo‘linadi** | O‘rta | Semantik sifat pasayadi | Multilingual embedding (default) + ixtiyoriy translation (M-T) |
| R3 | **Kichik LLM JSON buzadi** | Yuqori | Pipeline uziladi | Structured output + retry + graceful degrade |
| R4 | **Topic top terms embedding’da yo‘q** | O‘rta | Tahlil yuzasiga chiqmaydi | Embedding cluster’larini TF-IDF dan o‘tkazib top terms olish (M10) |
| R5 | **`min_df=2` vocabulary’ni bo‘shatadi** | O‘rta | Kichik PDF ishlamaydi | `min_df` ni chunk soniga moslash; `empty vocabulary` ni ushlash |
| R6 | **Ikki ustunli PDF matni aralashadi** | O‘rta | Chunk sifati pasayadi | `get_text("blocks", sort=True)`; cheklovni hujjatlashtirish |
| R7 | **Docker image juda katta (torch)** | Yuqori | Sekin build | CPU-only torch, slim base, layer cache |
| R8 | **Ollama modeli yuklanmagan** | O‘rta | `/health` 503 | `ollama pull` hujjati + startup tekshiruvi |
| R9 | **Scope kengayishi (RAG/DB/auth)** | O‘rta | Muddat cho‘ziladi | Non-Goals ro‘yxatini saqlash |
| R10 | **Embedding model 256 token chegarasi** | O‘rta | Uzun chunk’lar kesiladi | Chunk o‘lchamini `max_seq_length` ga moslash (M9) |
| R11 | **Tarjima terminologiyani buzadi** | O‘rta | Tahlil sifatiga ta’sir | Default translation **o‘chirilgan**; faqat M17 dan keyin yoqish |
| R12 | **Public deployda GPU/Ollama scaling** | O‘rta | Xarajat oshadi | Pluggable LLM provider — cloud’ga o‘tish config bilan |

---

# 17. Trade-Offs

| # | Qaror | Tanlangan | Rad etilgan | Sabab |
|---|-------|-----------|-------------|-------|
| T1 | Topic topish usuli | Unsupervised ML (K-Means) | LLM bilan mavzu berish | O‘lchanadi, takrorlanadi, halol |
| T2 | Vektorlash | TF-IDF **va** Embedding | Faqat bittasi | Ikki xil taqdimot — qo‘shimcha kuzatuv |
| T3 | Clustering algoritmi | K-Means | HDBSCAN / agglomerative | Tushunarli, keng o‘rgatilgan, `k` ni aniq boshqarish mumkin |
| T4 | Tarjima default’i | **O‘chirilgan** (Noop) | Doimiy yoqilgan | Local-first, xarajat, maxfiylik |
| T5 | Tarjima hajmi | document / page | chunk level | Kontekst yo‘qolmasligi, terminologiya, tezlik |
| T6 | Translation provider | Pluggable interface | Hardcoded Google | Kelajakda LibreTranslate/DeepL/Azure oson |
| T7 | LLM default | Ollama (lokal) | OpenAI default | Xarajat + maxfiylik; provider pluggable |
| T8 | Vektorlash o‘lchami | Stateless | DB | Soddalik; natija foydalanuvchida |
| T9 | Pipeline til bilan bog‘liq emas | Multi-strategy (TF-IDF yoki embedding) | Faqat TF-IDF | Embeddings semantik afzal |
| T10 | Chunk o‘lchami | Paragraf-asosida, 300–1200 | Fixed window | Semantik chegaralar tabiiy |
| T11 | Erorlar | Maxsus exception sinflari | `Exception` umumiy | Aniq xato xaritasi (400/413/422/503) |
| T12 | Fronteynd | Streamlit | React/Vue | Tez, Python-ekologiya, MVP uchun yetarli |

---

# 18. Translation Strategy (M-T)

**Yetakchi modul, alohida ishlov beriladi.**

## 18.1 Tamoyil

```text
Tarjima — MAJBURIY EMAS. Config orqali boshqariladi.
Default: TRANSLATION_ENABLED=false → NoopTranslator (passthrough, lokal).
```

## 18.2 Ikki rejim

| Rejim | Sozlama | Pipeline |
|-------|---------|----------|
| **A. Asl matn (default)** | `TRANSLATION_ENABLED=false`<br>`USE_MULTILINGUAL=true` | Multilingual embedding + `stop_words=None` |
| **B. Inglizga tarjima** | `TRANSLATION_ENABLED=true`<br>`TRANSLATION_PROVIDER=google_translate` | Monolingual embedding + `stop_words="english"` |

## 18.3 Qat’iy qoidalar

1. **Hech qachon chunk level tarjima qilinmaydi.** Faqat `document` yoki `page`.
2. **Tarjima pre-chunk** — sahifa darajasida bajariladi, keyin normal chunking.
3. **Default `Noop`** — tashqi so‘rov umuman yuborilmaydi.
4. **Graceful** — provider ishlamasa `TranslationError`/`TranslationUnavailableError` (→ HTTP 503).
5. **Interface-first** (`base.py` + Factory) — yangi provider qo‘shish `base.py` ni tegilmasdan.

## 18.4 Arxitektura (Strategy + Factory)

```text
app/translation/
├── __init__.py           public API
├── base.py               TranslatorProtocol / BaseTranslator
├── factory.py            get_translator() → config asosida
├── noop.py               NoopTranslator (default)
├── google_translate.py   GoogleTranslateTranslator (optional)
└── exceptions.py         TranslationError, TranslationUnavailableError
```

## 18.5 Config

```env
TRANSLATION_ENABLED=false
TRANSLATION_PROVIDER=none          # none | google_translate
TRANSLATION_TARGET=en
TRANSLATION_BATCH_BY=page          # document | page
USE_MULTILINGUAL=true
EMBEDDING_MULTILINGUAL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

> **Sabab:** aralash tilli PDF’da cluster’lar til bo‘yicha bo‘linishi mumkin. Multilingual embedding bitta vektor fazosida solishtiradi; tarjima esa monolingual pipeline’ni toza qiladi. Qaysi biri yaxshi — **M17’da o‘lchanadi**, hozir default — arzon va halol yo‘l.

---

# 19. LLM Strategy

| Qoida | Tushuntirish |
|-------|--------------|
| LLM = **explainer** | Mvzu/cluster topmaydi — faqat topilgan tuzilmani nomlaydi va tushuntiradi. |
| **Structured JSON** | `format=json` + Pydantic validation. |
| **Retry** | Yaroqsiz JSON → 1–2 marta qayta urinish. |
| **Kontekst chegarasi** | Butun PDF emas — faqat topic bo‘yicha qisqa kontekst (~500–800 belgi/chunk). |
| **Prompt’da qat’iy talab** | “Faqat berilgan matnga tayan, o‘ylab topma.” |
| **Til boshqaruvi** | `output_language` parametri. |
| **Graceful degrade** | Xato → ML natijasi qaytariladi, LLM maydonlari `None`. |
| **Pluggable** | `LLM_PROVIDER=ollama \| openai \| azure \| groq \| vllm`. |

---

# 20. Tasdiqlash

| # | Element | Holat |
|---|---------|-------|
| 1 | Scope, Goals, Non-Goals | ✅ M0.T1 |
| 2 | Requirements (F/NFR) | ✅ M0.T1 / M0.T2 |
| 3 | Constraints | ✅ M0.T1 |
| 4 | Architecture, Components, Data Flow | ✅ M0.T1 |
| 5 | API Boundaries, Data Storage | ✅ M0.T1 / M0.T2 |
| 6 | Failure Modes, Security, Scalability, Observability | ✅ M0.T1 |
| 7 | Deployment, Risks, Trade-Offs | ✅ M0.T1 |
| 8 | Translation Strategy | ✅ M0.T1 |
| 9 | LLM Strategy | ✅ M0.T1 |

**Keyingi qadam:** M0.T3 — namuna PDF’larni tayyorlash → M1 — muhit.
---

# 21. Architecture Summary

Unsupervised-first PDF topic analyzer. PyMuPDF → optional translation → preprocessing →
chunking → TF-IDF/K-Means (+ embeddings alternativasi) → topic discovery → LLM izoh →
FastAPI → Streamlit. Bitta monolith, alohida UI konteyneri, lokal LLM default.

**Asosiy qaror:** ML mavzularni topadi, LLM faqat tushuntiradi. LLM ishlamasa ham
pipeline natijasi saqlanadi (`topics` hech qachon bo'sh emas).

# 22. Database Strategy

**Ma'lumotbazasi yo'q.** Loyiha lokal, bir foydalanuvchiga mo'ljallangan.

| Nima | Qayerda | Hayot |
|------|---------|-------|
| PDF fayl | `data/raw/` (`.gitignore`) | M17 davomida |
| Keshlangan matn/chunk | `data/processed/` (`.gitignore`) | Kesh sifatida, o'chiriladi |
| Model (embedding) | HF cache, `models/` | Doimiy |
| Natijalar | `AnalysisResult` (JSON, API qaytaradi) | Vaqtinchalik, saqlanmaydi |

> Kelajakda (11-bo'limda rejalashtirilgan): natijalarni saqlash kerak bo'lsa —
> fayl (JSON/SQLite) yetarli. **DB qo'shish ADR talab qiladi** (ADR-006+).

# 23. Reliability

| Komponent | Ishlamasa | Natija |
|-----------|-----------|--------|
| Ollama (LLM) | `OllamaUnavailableError` → 503 | **ML natijasi saqlanadi**, `summary`/`key_points` = `None` |
| Translation provider | `TranslationUnavailableError` → 503 | **ML natijasi saqlanadi** (asl matn bilan) |
| Internet (tarjima/o'chish) | default `Noop` → tashqi so'rov yo'q | Pipeline **ishlaydi** |
| Model yuklash (lifespan) | startup xato | `/health` 503, aniq xato |

**Kafolatlar:** tashqi timeout **majburiy** (Ollama 120s, tarjima 15s);
retry **cheklangan** (2 marta, exponential backoff, infinite loop **yo'q**);
`random_state=42` → **deterministik**.

# 24. ADRs

| ADR | Qaror |
|-----|--------|
| [ADR-001](architecture/ADR-001-kmeans-over-bertopic.md) | TF-IDF + K-Means, BERTopic emas |
| [ADR-002](architecture/ADR-002-llm-local-ollama-pluggable.md) | Ollama default, pluggable LLM |
| [ADR-003](architecture/ADR-003-translation-pluggable-page-level.md) | Pluggable translation, page/document level |
| [ADR-004](architecture/ADR-004-cpu-only-torch-python-312.md) | Python 3.12 + CPU-only torch |
| [ADR-005](architecture/ADR-005-single-container-monolith.md) | FastAPI monolith + alohida Streamlit |

# 25. Implementation Instructions

Har implementator **majburiy** ravishda amal qiladi:

1. `opencode/AGENTS.md` §34 — `AGENTS.md` ni o'qishdan boshlang.
2. 8 bosqichli workflow (README §2.2) — `Plan` tugamaguncha kod yozma.
3. Har ML komponentida 10-qadamli o'rganish sikli (README §2.3).
4. Notebooks'da tajriba → `app/` ga ko'chir → test yoz.
5. Kontrakt buzmasin: `docs/io_contract.md`.
6. Test yozganingda **yurgiz** (`qa §59`: testni o'zgartirib o'tkazma).
7. Tugatishda majburiy hisobot (README §2.2.8 — 10 sarlavha).

# 26. Open Questions

| # | Savol | Kim hal qiladi | Qachon |
|---|-------|----------------|--------|
| 1 | Ollama yoki cloud LLM default bo'lsinmi (ADR-002 ko'ra yana baholash)? | Foydalanuvchi | M11 |
| 2 | Tarjima uchun Google yetarlimi yoki lokal LLM qo'shilsinmi (ADR-003)? | Foydalanuvchi | M-T |
| 3 | Embedding model: `qwen3` embeddinglar bilan bog'lash kerakmi? | M9 natijasi | M17 |
| 4 | Natija saqlash (history) kerakmi? → DB ADR | Foydalanuvchi | M18 |

> **OPEN DECISION** — 1-savol hali yopilmagan; `docs/io_contract.md` ikkala variantni ham qo'llab-quvvatlaydi.

# 27. Approval Required

| Qaror | Kim tasdiqlaydi | Holat |
|-------|-----------------|-------|
| Arxitektura (5 ta ADR) | Foydalanuvchi | ✅ **Tasdiqlangan** (M0) |
| Python 3.12 + CPU-only | Foydalanuvchi | ✅ **Tasdiqlangan** (M1) |
| Tarjima default ochiq | Foydalanuvchi | ✅ **Tasdiqlangan** (M0) |
| LLM default (Ollama vs cloud) | Foydalanuvchi | ⏳ **OPEN** — 26/1 |
| DB qo'shish | Foydalanuvchi | ⏳ Reja yo'q |

---

## Architecture Acceptance Criteria (`system-architect.md` §49)

- [x] Requirements understood
- [x] Non-functional requirements identified
- [x] System boundaries defined
- [x] Components have clear responsibilities
- [x] Dependencies understood
- [x] Data ownership defined (`docs/io_contract.md`)
- [x] API boundaries defined (`docs/io_contract.md` §2)
- [x] Failure modes considered (21-bo'limda Reliability)
- [x] Security boundaries considered (12-bo'lim)
- [x] Scalability risks considered (13-bo'lim)
- [x] Observability considered (14-bo'lim)
- [x] Deployment considered (15-bo'lim)
- [x] Major trade-offs documented (17-bo'lim + 5 ta ADR)
- [x] Unnecessary complexity rejected (monolith-first, ADR-005)
