# Cheklovlar (Limitations)

Bu hujjat **tekshirilgan** cheklovlarni yig'adi. Har bir yozuv
`tekshiruv` qismida qanday dalil bilan aniqlanganini ko'rsatadi —
hypothesis emas, o'lchov.

M2 da boshlangan; M17 da kengaytiriladi.

---

## L-01 — `get_text("sort=True")` ikki ustunli matnni buzadi

**Holat:** Ko'p ustunli (ikki ustunli akademik maqola, jadval, broshyura)
PDF'larida matn tartibi buzilishi mumkin.

**Tekshiruv:** Ikki hujjat yaratildi va ikkala variantda ham matn tartibi
solishtirildi.

| Hujjat | content stream | `sort=False` (standart) | `sort=True` |
|--------|-----------------|--------------------------|-------------|
| A — ustunlar ketma-ket yozilgan (real hujjatga o'xshash) | chap → o'ng | ✅ **to'g'ri** | ❌ qatorlab aralashgan |
| B — qatorlar aralash yozilgan | L0 R0 L1 R1 … | ❌ aralash | ❌ aralash |

**Xulosa:** `sort=True` matnni **vertikal koordinat** bo'yicha saralaydi,
shuning uchun ikki ustunni qatorlab aralashib ketiradi. U A holatda
**ishlaydigan yagona variantni buzadi** va B holatda ham yordam bermaydi.

**Qaror:** `app/services/pdf_extractor.py` da `sort=True` **ishlatilmaydi** —
standart `get_text("text")` (content stream tartibi) qo'llaniladi.
Regressiya testi: `tests/test_pdf_extractor.py::test_extract_pages_preserves_column_reading_order`.

**Keyingi qadam:** haqiqiy echim — **ustun aniqlash** (blok bbox'larini
klasterlash va ustun bo'ylab o'qish). Buni M4 da keram bo'lsa kiritiladi;
hozir yo'q.

> ⚠️ Bu README'dagi eski maslahatni **bekor qiladi**: "ikki ustunli PDF'da
> matn tartibi buziladi → `get_text("blocks", sort=True)` ni sinab ko'r".

---

## L-02 — Skanerlangan PDF qo'llab-quvvatlanmaydi

**Holat:** Rasmdan skanerlangan PDF'ning matn qatlami yo'q.

**Tekshiruv:** `data/raw/` ga rasmli PDF yaratilib, `extract_pages()`
chaqirildi → `ScannedPdfError` (0 belgi < 100 chegara). Notebook 01 §5.

**Qaror:** OCR — **non-goal** (`docs/io_contract.md` §1.3). Xato aniq
beriladi, ML pipeline ishga tushirilmaydi.

**Oqibat:** Foydalanuvchi qo'lda skanerlash va qayta yuklash kerak.
Agar bu talab kuchayib ketsa, OCR — alohida milestone bo'lishi kerak.

---

## L-03 — Parol bilan himoyalangan PDF qo'llab-quvvatlanmaydi

**Holat:** `needs_pass == True` bo'lgan PDF.

**Tekshiruv:** AES-256 bilan shifrlangan PDF yaratildi. `pymupdf.open()`
**muvaffaqiyatli**, lekin matn bo'sh qaytaradi.

**Nega alohida tekshiriladi:** parol bilan va skanerlangan holatlar
**bir xil ko'rinadi** (ikkalasida ham belgi = 0). Agar `needs_pass`
tekshirilmasa, parol bilan himoyalangan PDF "skanerlangan" deb xato
baholanadi — foydalanuvchi noto'g'ri maslahat oladi.

**Qaror:** `InvalidPdfError` (i/o contract §1.2, I5).

---

## L-04 — `MIN_TOTAL_CHARS` chegarasi ikki tomonlama xato qiladi

**Holat:** chegara — kompromiss.

**Tekshiruv:** 40 belgili o'qilgan PDF `ScannedPdfError` beradi (100
chegarasidan kam). 2 000+ belgili skanerlangan PDF esa chegaradan
o'tib, ML pipeline'ga kirib, chalkash natija beradi.

**Qaror:** 100 (`Settings.min_total_chars`, `.env` dan o'qiladi).
Past chegara oddiy PDF'ni rad etadi, yuqori chegara skanerlangan PDF'ni
o'tkazib yuboradi.

**Keyingi qadam:** o'rtacha chegaradan pastroq, lekin o'zgaruvchan
(`sahifalar_soni * 50` kabi) qilish mumkin — M13 da haqiqiy foydalanuvchi
ma'lumotlari asosida.

---

## L-05 — Testlar `data/raw/` ga bog'liq emas

**Holat:** `data/raw/*.pdf` `.gitignore`da — katta PDF'lar git'ga kirmaydi.

**Tekshiruv:** klonlangan repo'da bu fayllar **yo'q**. Agar testlar
ulardan foydalansa, `pytest` klon keyin darhol qatlaydi.

**Qaror:** `tests/test_pdf_extractor.py` barcha PDF'larni `pymupdf` bilan
test ichida yaratadi. Namuna PDF'lar faqat **notebook** va qo'lda
tekshirish uchun (`scripts/make_sample_pdfs.py`).

**Oqibat:** Test muhiti tashqi fayllarga bog'liq emas — CI uchun tayyor.

---

## L-06 — `insert_textbox` matnni kesadi (test helper cheklovi)

**Holat:** PDF yaratishda `insert_text` sahifa chegarasidan tashqariga
chiqqan matnni **kesib** qo'yadi — `get_text` qisqarilgan matnni
qaytaradi (belgilar 32 dan 95 ga tushdi, kutilgan 128 emas).

**Qaror:** test helper'ida `insert_textbox` + rect ishlatiladi — matn
o'raydi va to'liq saqlanadi.

**Darslik:** test fixture yaratishda ham "juda oson" xato bo'ladi —
test yashirib qo'ygan real muammo.

---

## L-07 — `sklearn` ning `ENGLISH_STOP_WORDS` texnik so'zlarni o'ldiradi

**Holat (o'lchov):** `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`
318 so'zdan iborat ro'yxat. U **umumiy ingliz** uchun yozilgan,
texnik korpus uchun emas. 37 ta muhim domen so'zi tekshirildi —
ulardan **`system`** shu ro'yxatda bor. Bizning namuna
`02_medium_distributed_systems.pdf` esa aynan shu mavzuda.

**Oqibat:** to'g'ridan-to'g'ri ishlatilsa, mavzu so'zi lug'atdan
o'chib ketadi va klasterlash **noto'g'ri** natija beradi.

**Qaror:** `STOP_WORDS = ENGLISH_STOP_WORDS - PROTECTED_TERMS`.
`app/nlp/preprocessing.py` da 100+ so'zli `PROTECTED_TERMS` kiritilgan.
Regressiya bo'lmasligi uchun test bor:
`test_no_protected_term_leaks_into_stop_words`.

**Darslik:** tayyor kutubxona ro'yxati — boshlanish nuqtasi, yakuniy
javob **emas**. Ishlatishdan oldin o'z korpusingiz bilan tekshirish
shart.

---

## L-08 — Namuna PDF'larimiz dekorativ chiziq va boshqa artefaktlarni **yo'qotadi**

**Holat (o'lchov):** 4 ta namuna PDF tekshirildi
(`notebooks/02_preprocessing.ipynb` §1). Topilgan artefaktlar:

| Artefakt | Namuna PDF'larda |
|----------|-------------------|
| Takrorlanuvchi header | ✅ 4/4 faylda |
| `Page N of M` footer | ✅ 4/4 faylda |
| `=` dekorativ chiziq | ✅ 4/4 faylda (3–12 marta) |
| Defis bilan uzilgan so'z | ❌ **yo'q** |
| Ligatura (`ﬁ`, `ﬂ`) | ❌ **yo'q** |
| `\xa0` (no-break space) | ❌ **yo'q** |

**Oqibat:** jami belgining 12–17% i artefakt edi va olib tashlandi.
Lekin uchta funksiya — `fix_hyphenation`, `normalize_unicode`,
bo'shliq normallashtirish — **faqat sintetik testlarda** tekshirildi.
Bizning generatorimiz (DejaVu/ReportLab) toza matn beradi.

**Qaror:** bu funksiyalar **haqiqiy namuna**da emas, sintez matnlarda
test qilinadi. "Real PDF'da tekshirildi" deb da'vo qilinmaydi.

**Darslik:** o'z generatorimiz qilib yaratgan ma'lumot — hujjatsiz
tekshiruv uchun yetarli emas. Keyin (M4/M5 da) haqiqiy manbalardan
PDF kiritish kerak.

---

## L-09 — Stop-word filtrining o'zi kichik o'lchovda ma'nosiz natija beradi

**Holat (o'lchov):** filtrdan o'tgan **72** token solishtirildi
(4 ta namuna PDF bo'yicha). Hammasi — haqiqiy ingliz function
so'zlari (`the, of, and, is, a, in, ...`). **Domen so'zi yo'qoldi**,
`k-means` va `tf-idf` kabi atamalar butunligicha saqlandi.

**Cheklov:** bu kichik namuna (4 hujjat). Katta haqiqiy korpusda
`system` ga o'xshash ammo o'z-o'zidan chiqishi mumkin. M5 da
lug'atdagi eng ko'p uchraydigan tokenlar qo'lda ko'rib chiqiladi.

---

## L-10 — Stemming strategiyasi o'lchangan, lekin Porter emas

**Holat:** M3 da uch strategiya solishtirildi. C (stemming) eng kichik
lug'atni berdi (854 → 775, 9.3%), lekin texnik atamalarni buzdi
(`kmeans → kmean`, `embedding → embedd`).

**Cheklov:** muhitda stemmer **yo'q** (`nltk` `requirements.txt` da
yo'q). O'lchovda ishlatilgan `crude_stem` — ataylab yozilgan sodda
qat'iyat kesuvchi, Porter algoritmi **emas**. Shu sababli C ning haqiqiy
natijasi boshqacha bo'lishi mumkin.

**Qaror:** default B (stemming yo'q). Haqiqiy stemmer talab bo'lsa,
M5 da `nltk` bog'liqligi alohida qaror bilan qo'shiladi.

---

## L-11 — Stop-word ro'yxati **faqat ingliz**; boshqa tillar tozalanmaydi

**Holat (o'lchov):** B strategiyasining top-10 token'ida **`va`**
turibdi — bu o'zbekcha "va" (and). Sabab: `STOP_WORDS` — ingliz
ro'yxatidan olingan, o'zbek va ruscha function so'zlari unda yo'q.
`04_mixed_language.pdf` da 561 ta Kirill belgisi bor, ular ham
tozalanmaydi.

**Oqibat:** aralash tilli korpusda `ml_text` sezilarli shovqin qoldiradi.
Bu `m3_probe` o'lchovida ham ko'rindi: A va B strategiyasining farqi
boshqa tilli hujjatda kamroq bo'ladi.

**Qaror:** Hozircha chetlab o'tiladi, chunki:
- loyihaning asosiy maqsadi **ingliz** texnik hujjatlar (README);
- ko'p tilli stop-word ro'yxatini qo'shish — yangi qaror va yangi
  bog'liqlik.

Ko'p tilli korpusda natija **yaxshi bo'lmaydi** — bu M5 da o'lchab
tasdiqlanishi kerak.

**Keyingi qadam:** M5 da top-10 token'lar qo'lda ko'rib chiqiladi va
kerak bo'lsa, til aniqlanadi (M5 + M6 = tilni belgilash kerak).

---

## L-12 — Haqiqiy kitobda `fix_hyphenation` so'z **yaratib** yuboradi

**Sintetik PDF'larda bu artefakt umuman yo'q edi** (L-08). Real
kitoblar (`data/raw/real/`) da satr oxirida defis bilan uzilgan
so'zning **ikki xil turi** bor:

| Tur | Namuna | To'g'ri natija |
|-----|--------|----------------|
| Bitta so'z sindi | `transduc-` + `tion` | `transduction` |
| Ikki so'z, defisli | `book-` + `shelves` | `book-shelves` |

Avvalgi implementatsiya `(\\w)-` bilan **har doim** birlashtirardi.
O'lchov — 10 kitob + 5 ochiq maqola, to'liq korpus
(5 501 150 belgi, 2 596 sahifa), **750** ta hyphenation nuqtasi:

| Guruh | Nuqta | Bitta so'z sindi | Haqiqiy kompaniya |
|-------|-------|------------------|-------------------|
| Adabiyot (2 527 sahifa) | 404 | 60 (15%) | **344 (85%)** — `book-shelves`, `cherry-tart`, `twenty-four`, `star-ﬁsh`, `rose-tree` |
| arXiv (39 sahifa) | 206 | 136 (66%) | 70 (34%) — `sequence-aligned`, `position-wise`, `source-target`, `English-to` |
| PLOS (30 sahifa) | 140 | 108 (77%) | 32 (23%) — `A-RC`, `Nose-Throat` |

Jo'shilmay birlashtirilsa, **446** ta (59%) haqiqiy kompaniya buzilardi
(`bookshelves`, `cherrytart`, `twentyfour`, `sequencealigned`).

**Yechim:** `build_vocabulary()` korpusdan tokenlar yig'adi va
`fix_hyphenation` faqat **korpusda tasdiqlangan** shakllarni
birlashtiradi.

**Nega bu xavfsiz yo'l** — xatolar **nomutanosib**:

| Xato | Natijasi | Yomonligi |
|------|----------|-----------|
| Keraksiz birlashtirish | `bookshelves` — bunday so'z **yo'q** | iflos token lug'atga kiradi |
| Birlashtirmaslik | `book-shelves` saqlanadi | bu ham **to'g'ri** token |

Yomonroq holatni yo'q qilish kerak, shuning uchun konservativ yo'l
tanlandi.

**Qoldiq cheklov:** so'z korpusda **hech qayerda butun** ko'rinmasa,
birlashtirilmaydi. Bu ataylab qoldirilgan — boshqa usul ishonchli
ro'yxat talab qiladi (`nltk` yo'q, yangi bog'liqlik qo'shilmadi).

---

## L-13 — Ligatura lug'atni ifloslantiradi (topildi va tuzatildi)

`dracula.pdf` da **1 691** ta ligatura, butun korpusda **11 594** ta:
`dracula.pdf` da `ﬁ` (830), `ﬀ` (424), `ﬂ` (298), `ﬃ` (123), `ﬄ` (16).

Bir kitobni tekshirish yetarli emas — adabiyot korpusida jami
**11 345** ta ligatura bor (11 ta kitob). Namuna asosida xulosa
yozish — "real ma'lumot bilan tekshirdim" degan narsa emas.

`build_vocabulary` dastlab `normalize_unicode` **siz** yig'ilardi.
Natija: lug'atda `suﬃcient` turardi, ammo solishtiriladigan shakl
`sufficient` — **hech qachon mos kelmasdi**, ya'ni birlashtirish
o'tkazib yuborilardi. `fix_hyphenation` ning lug'at bilan ishlangan
variantini yozishda bu xato darhol ko'rinib chiqdi.

**Tuzatildi:** `build_vocabulary` avval `normalize_unicode` qiladi.
Real korpusdan o'lchangan natija: ligatura **11 594 → 0**.

---

## L-15 — **Boshqa kitob** yuklangan: ID boshqaruv emas, dalil

Gutenberg ID `25438` "The Jungle" (Upton Sinclair) deb yozilgan edi.
Yuklangan fayl esa **"The Airlords of Han"** (Amazing Stories, 1929)
edi — sarlavha matnda **umuman yo'q** edi.

To'g'ri raqam — **140**. (`25438` — boshqa asar; xotiradan
taxmin qilish yetarli emas.)

**Nima sabab bo'ldi.** Kod to'g'ri edi, lekin **dalil** yo'q edi:
hech kim yuklangan kontentni e'lon qilingan sarlavha bilan
**solishtirmagan**. Natijada xato o'tkir ko'rinmaydi — PDF ochiladi,
sahifa bor, belgi bor, `pdf_stats()` muvaffaqiyatli qaytaradi. Faqat
**kontent boshqa**.

**Nima qilindi.** `scripts/fetch_real_documents.py` ga
`verify_title(text, title, author)` qo'shildi: har bir faylda
sarlavhaning ma'nosiz so'zlari (`The`, `A`, `An`, `of`) va muallifning
birinchi so'zi **matn ichidan** qidiriladi. Topilmasa — `FetchError`,
hujjat manifestga **kirmaydi**.

**Bu kod xatosi emas, ma'lumot xatosi** — va hech qanday birlik test
uni ushlamagan, chunki testlar tarmoqqa ulanmaydi (L-05). Yechim
ma'lumot chegarasida: `data/manifest.json` da har bir hujjat uchun
`sha256` va `title` saqlanadi, va `tests/test_real_corpus.py` ularni
tekshiradi.

**Oqibat:** korpus bir vaqtincha 178 KB noto'g'ri kitobni o'z ichiga
olgan edi; tuzatgandan keyin "The Jungle" — 808 KB. O'lchov raqamlari
shu sabab **ikki marta** qayta hisoblandi.

---

## L-14 — Chekka (`edge_lines`) real PDF da header ni topmaydi

PDF matn qatlami **vizual tartibni saqlamaydi**. `plos_middle_ear_effusion.pdf`
da header `PLOS ONE | https://doi.org/...`:

- 3-sahifada — matn **oxirida**
- 1-sahifada — matn **o'rta qatorda** (73 qatordan 35-36 da)

`edge_lines=3` shu sabab 3-sahifadagini topadi, o'rta qatordagini
**yo'q** qoldiradi.

`edge_lines=None` (hamma joyda izlash) topadi — lekin **juda xavfli**
(default `min_ratio=0.5` bilan o'lchangan):

| Fayl | `edge_lines=3` | `edge_lines=None` |
|------|----------------|-------------------|
| `arxiv_bradley_terry.pdf` | −1.3% | **−4.0%** |
| `plos_middle_ear_effusion.pdf` | −1.7% | −1.6% |

Olib tashlanganlar **furniture emas**, ilmiy mazmun edi: `X` (67 marta),
`i` (56), `1` (43), `\uf8f4` (38) — formula jadvali belgilari.
Jami **509 qator** yo'q qilindi.

**Xulosa:** `None` optsional, lekin **ishlatilmaydi** va
`clean_pages` undan foydalanmaydi. To'g'ri tuzatish — M2 dan blok
koordinatalarini (`get_text("blocks")`) olish va `y` joylashuviga qarab
filtrlash; bu M3 doirasida emas.

---

## Tekshirilmagan (gumon bo'lib qolgan)

| Gumon | Holat |
|-------|-------|
| Jadvallar (table) matnni to'g'ri ajratadimi? | ❌ tekshirilmagan — M4 da kerak |
| Rasmli ichki sahifalar (muqova, diagramma) | ❌ tekshirilmagan |
| Matrecal / MathType formulalar | ❌ tekshirilmagan |
| 100+ sahifali PDF (`docs/io_contract.md` §1.3 cheklovi) | ✅ tekshirilgan — 397 sahifali `dracula.pdf` |
| Nested/ichki PDF'lar (embedded) | ❌ tekshirilmagan |
| Jadvallardagi satr oxiridagi defis (jadvallarda `k-` + keyingi qator) | ❌ tekshirilmagan — M4 da |
| Ko'p tilli matnda `ml_text` (faqat ingliz stop-word) | ❌ cheklov — L-11 |
| Jadvallardagi takrorlanuvchi ustun sarlavhalari | ❌ cheklov — L-14 |
| Skanerlangan/rasm ko'rinishidagi real adabiyot (Gutenberg faqat matn beradi) | ❌ cheklov — `scripts/make_real_pdfs.py`