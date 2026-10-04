# Input / Output Contract — PDF Analyzer

**Version:** 1.0.0
**Status:** M0 — tasdiqlangan
**Manba:** `docs/architecture.md`, `README.md` (7-bo'lim, 8-bo'lim)

Bu hujjat **M2–M13** davomida yoziladigan barcha modullar uchun **kcontract** (shartnoma) hisoblanadi. Kod shu hujjatga mos bo'lishi shart.

---

# 1. Input Contract

## 1.1 Kirish nuqtalari

Pipeline bir necha joydan kirish qabul qiladi — barchasi **bir xil semantik** ga ega bo'lishi shart:

| Kirish | Tip | Kim beradi | Qayerda ishlatiladi |
|--------|-----|-----------|---------------------|
| Fayl yo'l | `str` (path) | Notebook, test, CLI | `extract_pages(path)` |
| Fayl bytes | `bytes` | FastAPI `UploadFile` | `extract_pages(bytes)` |
| PDF stream | `io.BytesIO` | Kelajak (ixtiyoriy) | `extract_pages(stream)` — `bytes` bilan bir xil semantika |

```python
def extract_pages(source: str | bytes) -> list[PageText]: ...
```

## 1.2 Input qoidalari

| # | Qoida | Tekshiruv | Xato |
|---|-------|-----------|------|
| I1 | Fayl **PDF** formatida | Magic bytes `%PDF` (birinchi 5 bayt) | `InvalidPdfError` |
| I2 | Fayl **bo'sh emas** | Hajm > 0 bayt | `InvalidPdfError` |
| I3 | Hajm `MAX_UPLOAD_MB` dan oshmasin | `len(source) <= MAX_UPLOAD_MB * 1024**2` | API: **413** |
| I4 | **Matn qatlami** mavjud | Jami belgi >= `MIN_TOTAL_CHARS` (100) | `ScannedPdfError` |
| I5 | **Parol bilan himoyalanmagan** | `doc.needs_pass == False` | `InvalidPdfError` |
| I6 | Sahifalar **o‘qilishi mumkin** | `pymupdf.open()` muvaffaqiyatli | `InvalidPdfError` |
| I7 | Kamida **1 sahifa** | `len(doc) >= 1` | `InvalidPdfError` |

> **I1 — eng muhim.** Kengaytma `.pdf` ga ishonish **yetarli emas**. Foydalanuvchi `test.pdf` nomli PNG yuborsa — `pymupdf` ochmaydi, lekin tekshiruv aniqroq xato beradi. Magic bytes — har doim tekshiriladi.

## 1.3 Input chegaralari (MVP)

| Cheklov | Qiymat | Sabab |
|---------|--------|-------|
| Fayl turi | faqat `.pdf` | MVP scope |
| Hajm | <= 25 MB (`MAX_UPLOAD_MB`) | Xotira + vaqt |
| Sahifalar | <= ~100 (boshlang'ich) | Vaqt; keyinchalik oshiriladi |
| Tillar | har qanday (multilingual pipeline) | M-T |
| Chalkashlik (encryption) | ❌ qo'llab-quvvatlanmaydi | M2 |
| Skanerlangan (rasm) | ❌ qo'llab-quvvatlanmaydi (OCR yo'q) | Non-goal |

## 1.4 Validatsiya qayerda?

| Validatsiya | Qatlam | Mas'ul |
|-------------|--------|--------|
| Magic bytes, hajm, content_type | **API** | M13 |
| `needs_pass`, oqilishi, matn borligi | **PDF service** | M2 |

> **Ikkalasi ham kerak.** API — foydalanuvchi uchun tez, aniq xato (400/413). Service — API bo'lmaganda ham (notebook, test) himoya.

---

# 2. Output Contract

## 2.1 Ichki modellar (M2/M4)

### `PageText` — M2

```python
class PageText(BaseModel):
    page: int      # 1-dan boshlanadi (odamga qulay)
    text: str      # xom matn (bo'sh sahifalar ham saqlanadi)
```

**Qoidalar:**
- `page` — **1-based** (foydalanuvchi ko'radigan raqam).
- Bo'sh sahifalar (`text == ""`) **o'chirilmaydi** — statistika uchun saqlanadi.
- `text` — `.strip()` qilingan.

### `Chunk` — M4

```python
class Chunk(BaseModel):
    id: int                # 0-dan boshlanadi, monoton
    text: str              # display_text (chunk'lash uchun)
    pages: list[int]       # qaysi sahifa(lar)dan (1-based, unique, sorted)
    n_chars: int           # len(text)
```

**Qoidalar:**
- `id` — indeks, uzluksiz (`0..n-1`).
- `pages` — **bo'sh bo'lmasligi shart** (har chunk kamida 1 sahifadan).
- `pages` — unique + sorted.
- `n_chars` — `MIN_CHUNKS_FOR_CLUSTERING` ga bog'liq emas, lekin M4 da `MIN_CHARS`/`MAX_CHARS` ga rioya qilinadi.

### `Topic` — M10

```python
class Topic(BaseModel):
    cluster_id: int
    size: int                       # cluster'dagi chunk soni
    share: float                    # size / total_chunks, 0..1
    top_terms: list[str]            # 8-10 ta so'z
    representative_chunks: list[int]  # chunk id'lar (3 ta)
    pages: list[int]                # cluster qamrab olgan sahifalar
    label: str | None = None        # M11 (LLM) to'ldiradi
    description: str | None = None  # M11 (LLM) to'ldiradi
```

**Qoidalar:**
- `share` — `0 < share <= 1`, `sum(topic.share for topic in topics) == 1.0` (float toleransiya bilan).
- `top_terms` — **bo'sh bo'lmasligi shart** (embedding rejimida ham TF-IDF orqali olinadi).
- `representative_chunks` — `cluster_id` ga tegishli chunk id'lar.
- `label`/`description` — ML bosqichida **`None`**; LLM ishlamasa ham **`None`** qoladi (graceful).

## 2.2 API javob modeli (M13)

```json
{
  "document": "paper.pdf",
  "language": "uz",
  "translated": false,
  "representation": "embedding",
  "k": 5,
  "stats": {
    "n_pages": 24,
    "n_chunks": 87,
    "silhouette": 0.084,
    "inertia": 412.7,
    "processing_time_sec": 31.2
  },
  "main_topic": "Asosiy mavzu nomi",
  "topics": [
    {
      "cluster_id": 0,
      "label": "Neural Networks",
      "description": "Bu cluster ... haqida to'plangan chunk'lardan iborat.",
      "size": 24,
      "share": 0.276,
      "top_terms": ["neural", "network", "layer", "training"],
      "representative_chunks": [3, 17, 42],
      "pages": [2, 3, 4, 8]
    }
  ],
  "key_points": ["...", "...", "..."],
  "important_sections": [
    { "pages": [2, 3], "chunk_ids": [3, 4, 5], "reason": "cluster 0 vakili" }
  ],
  "summary": "Umumiy xulosa...",
  "warnings": ["LLM mavjud emas — topic label'lar to'ldirilmadi"]
}
```

### Manda'tiy (required) maydonlar

| Maydon | Turi | Izoh |
|--------|------|------|
| `document` | `str` | Fayl nomi |
| `main_topic` | `str \| None` | LLM bo'lmasa `None` |
| `topics` | `list[Topic]` | **Har doim bo'ladi** (ML natijasi) |
| `key_points` | `list[str]` | LLM bo'lmasa `[]` |
| `summary` | `str \| None` | LLM bo'lmasa `None` |
| `stats` | `object` | Diagnostika uchun |

### Ixtiyoriy (optional) maydonlar

| Maydon | Default | Izoh |
|--------|---------|------|
| `language` | `None` | Aniqlangan til (M17 dan keyin) |
| `translated` | `false` | Tarjima ishlatilganmi |
| `representation` | `"embedding"` | `"tfidf"` \| `"embedding"` |
| `k` | `None` | Tanlangan cluster soni |
| `important_sections` | `[]` | Muhim bo'limlar (sahifa bilan) |
| `warnings` | `[]` | Graceful degradation xabarlari |

> **Muhim:** `topics` **hech qachon bo'sh qolmaydi** (LLM yo'q bo'lsa ham). Faqat LLM maydonlari `None`/`[]` bo'ladi. Bu — M12 graceful degradation ning asosiy kafolati.

## 2.3 Warnings (nima uchun, qachon)

| Warning | Sabab |
|---------|-------|
| `"LLM mavjud emas — topic label'lar to'ldirilmadi"` | Ollama ishlamadi (M11) |
| `"LLM javobi validatsiyadan o'tmadi"` | Noto'g'ri JSON (retry dan keyin) |
| `"Tarjima provider ishlamadi — asl matn bilan tahlil qilindi"` | Translation fallback (M-T) — ML natijasi saqlanadi |
| `"Katta PDF — tahlil uzoq davom bo'lishi mumkin"` | > 50 sahifa |

**Warnings — foydalanuvchi uchun tushunarli tilda.** Ichki tafsilot (traceback, URL, API key) **chiqmaydi**.

---

# 3. Baholash mezonlari (M7, M17)

## 3.1 Nima uchun ikki xil baholash?

Unsupervised learning'da **"to'g'ri javob" (ground truth) yo'q**. Shuning uchun:

```text
Quantitative (raqam)  +  Qualitative (qo'lda ko'rish)  =  ishonchli baholash
```

Faqat raqam — "chiroyli", lekin ma'nosiz natija berishi mumkin.
Faqat ko'z — subyektiv, takrorlanmaydi.

## 3.2 Quantitative mezonlar

| # | Metrika | Qayerda | Talqin | Yaxshi belgi |
|---|---------|---------|--------|--------------|
| Q1 | **Silhouette** (cosine) | M7 | `-1..1`; ichki zichlik + ajralganlik | `> 0.05` signal; `< 0` yomon |
| Q2 | **Inertia** | M6/M7 | Ichki kvadrat masofa yig'indisi | Kichik, lekin `k` oshsa **doim** kamayadi |
| Q3 | **Cluster o'lchamlari** | M6 | `min`, `max`, taqsimot | Bir cluster > 70% yoki < 3 chunk = signal |
| Q4 | **Stability** (5 seed) | M7 | Seed o'zgarganda o'xshashmi? | Minimal o'zgarish |
| Q5 | **Adjusted Rand Index** | M7 (ixtiyoriy) | Seed'lar orasida o'xshashlik | 1.0 = bir xil |
| Q6 | **Elbow** | M7 | Inertia egri chizig'i | Aniq "tirsak" nuqtasi |

> **Q1 haqida ogohlantirish.** Matnli (sparse) ma'lumotda silhouette **mutlaq qiymat** past bo'ladi — 0.02–0.15 odatiy. Bu "yomon" degani emas. **Qiymat emas, TREND va boshqa metrikalar bilan solishtirish** muhim.

### `k` qanday tanlanadi?

```text
1. k_max = min(K_MAX, n_chunks // K_DIVISOR)     # K_DIVISOR = 5
2. k = 2..k_max oralig'ida evaluate_k_range()
3. Eng yuqori silhouette -> "tavsiya" (heuristic)
4. LEKIN yakuniy qaror = RAQAM + QO'LDA KO'RISH
5. Qaror docs/evaluation_report.md'ga yoziladi (sabab bilan)
```

> **Bitta "to'g'ri" `k` yo'q.** Silhouette — tavsiya, xulosa emas.

## 3.3 Qualitative mezonlar (manual inspection rubric)

Har bir cluster uchun **1–5** ball. M17 da to'ldiriladi.

| # | Mezon | 1 (yomon) | 3 (o'rtacha) | 5 (yaxshi) |
|---|-------|-----------|--------------|------------|
| **K1 — Izchillik** (Coherence) | Chunk'lar butun boshqa mavzular | Aralash | Barchasi bir mavzu |
| **K2 — Aniqlik** (Clarity) | Top terms mavzuni aks ettirmaydi | Qisman | Aniq ifodalagan |
| **K3 — Farqlanish** (Separation) | Boshqa cluster'dan farq qilmaydi | Qisman | Aniq ajratilgan |

**Ish tartibi (M17):**

```text
1. Cluster o'lchamlari     → bitta >70% yoki <3 chunk'li bormi?
2. Silhouette (cosine)     → >0.05 signal; <0 yomon belgi
3. Elbow                   → egilish nuqtasi bormi?
4. Stability               → 5 seed → o'xshash cluster'larmi?
5. Top terms               → mavzuni ifodalaydimi?
6. Representative chunks   → qo'lda o'qi: bir mavzumi?
7. QAROR                   → raqam + kuzatuv bilan asosla, docs'ga yoz
```

## 3.4 Qizil bayroqlar (red flags)

| # | Belgi | Nimani anglatadi | Qayerga qaytish |
|---|-------|-----------------|-----------------|
| RF1 | Top terms: `the, of, and, figure, page` | Tozalash yoki `max_df` yetarli emas | M3, M5 |
| RF2 | Har cluster = ketma-ket sahifalar | Bu **mavzu emas**, hujjat strukturasi | M4, M5, M9 |
| RF3 | Bitta cluster 80–90% | Vektorlash yoki `k` muammoli | M5, M6, M7 |
| RF4 | Silhouette < 0 | Clusterlar chalkash yoki k juda kichik | M7 |
| RF5 | Representative chunk'lar mavzusiz | Top terms representative'ga mos emas | M10 |
| RF6 | LLM label'leri matnda yo'q narsani aytadi | **Hallucination** — prompt'ni kuchaytirish | M11, M17 |

> **RF2 — eng ko'p uchraydigan xato.** Chunk'lar sahifa tartibida yig'ilgani uchun K-Means "mavzu" o'rniga "hujjat qismi" topishi mumkin. Yechim: chunk aralashligini oshirish yoki embedding'ga o'tish.

## 3.5 Baholash jadvali shakli (M17)

Har bir PDF × har bir representation uchun:

| PDF | repr | k | silhouette | inertia | min_size | max_size | stability | K1 | K2 | K3 | Xulosa |
|-----|------|---|-----------|---------|----------|----------|-----------|----|----|----|--------|
| paper.pdf | tfidf | 5 | 0.071 | 412.7 | 8 | 24 | 0.82 | 4 | 4 | 3 | yaxshi |
| paper.pdf | embedding | 5 | 0.084 | 88.1 | 6 | 22 | 0.91 | 5 | 4 | 4 | **yaxshiroq** |
| book.pdf | tfidf | 8 | 0.043 | 1204.0 | 2 | 89 | 0.55 | 2 | 3 | 2 | RF3 bor |
| book.pdf | embedding | 6 | 0.092 | 201.4 | 9 | 41 | 0.78 | 4 | 5 | 4 | yaxshi |

## 3.6 Translation ta'sirini baholash (M-T + M17)

| PDF | Rejim | repr | k | silhouette | K1-K3 | Xulosa |
|-----|-------|------|---|-----------|-------|--------|
| mixed.pdf | A: asl (multilingual) | embedding | 4 | 0.061 | 3/3/2 | cluster'lar **til bo'yicha** bo'lingan (RF2) |
| mixed.pdf | B: tarjima (EN) | tfidf | 4 | 0.089 | 4/4/4 | mavzu bo'yicha toza ajratilgan |

**Xulosa qoidasi:** B rejim yaxshi bo'lsa ham, **default A qoladi** (local-first, arzon, maxfiylik). B faqat **kerak bo'lganda** yoqiladi. Qaror `docs/evaluation_report.md`'da sabab bilan yoziladi.

---

# 4. Reproducibility shartlari

| # | Shart | Qayerda |
|---|-------|---------|
| R1 | `random_state=42` (`RANDOM_SEED`) | Barcha ML qadamlar |
| R2 | Tasodifiy chunk tartibi yo'q | M4 |
| R3 | Model keshi (`_model` global) — bir marta | M9 |
| R4 | `k` avtomatik tanlanganda seed qotirilgan | M7/M12 |
| R5 | LLM natijasi — har doim emas (temperature 0) | M11 |

> **Test:** bir xil `random_state` → **bir xil** label'lar (M15 test).

---

# 5. Xato xaritasi (qisqacha)

| Xato | HTTP | Qayerda |
|------|------|---------|
| `InvalidPdfError` | 400 | M2 |
| Hajm katta | 413 | M13 |
| `ScannedPdfError` | 422 | M2 |
| `NotEnoughContentError` | 422 | M4/M12 |
| `OllamaUnavailableError` | 503 (+ ML natijasi) | M11/M13 |
| `TranslationUnavailableError` | 503 (ML natijasi ham qaytariladi) | M-T/M13 |
| `TranslationError` (noto'g'ri config) | 500 | M-T/M13 |
| Kutilmagan | 500 (ichki tafsilot **yo'q**) | M13 |

---

# 6. Kelajakda o'zgarishi mumkin (MVP chegarasi)

| # | O'zgarish | Qachon |
|---|-----------|--------|
| 1 | `topics[]` ga `human_label` qo'shiladi | M17 dan keyin |
| 2 | `stats` ga `topic_ari` qo'shiladi | M17+ |
| 3 | `language` — avtomatik aniqlanadi | M17+ |
| 4 | `document_id` — natija saqlanadi | Non-goal (DB) |
| 5 | Streaming `key_points` | Non-goal |

> Ushbu o'zgarishlar **backward-compatible** bo'lishi shart — ixtiyoriy maydonlar qo'shiladi, mavdonlar o'chirilmaydi.