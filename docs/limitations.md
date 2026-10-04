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

## M2 da tekshirilmagan (gumon bo'lib qolgan)

| Gumon | Holat |
|-------|-------|
| Jadvallar (table) matnni to'g'ri ajratadimi? | ❌ tekshirilmagan — M4 da kerak |
| Rasmli ichki sahifalar (muqova, diagramma) | ❌ tekshirilmagan |
| Matrecal / MathType formulalar | ❌ tekshirilmagan |
| 100+ sahifali PDF (`docs/io_contract.md` §1.3 cheklovi) | ❌ tekshirilmagan |
| Ligatura (`ﬁ`, `ﬂ`) va maxsus belgilar | ❌ M3 da (README §M3) |
| Nested/ichki PDF'lar (embedded) | ❌ tekshirilmagan |