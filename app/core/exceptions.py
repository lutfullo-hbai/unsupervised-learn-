"""Loyihaning domain xatolari — bitta ildizdan (`docs/io_contract.md` §4).

Nega bitta ildiz? API qatlami (M13) har bir xotani alohida ushlab, unga
HTTP kodini biriktirishi kerak. Agar har modul o'z xatosini yaratsa,
`except` zanjiri tarqaladi va yangi modul qo'shilganda API qatlamini
o'zgartirish kerak bo'ladi. Bir ildiz — **bir marta** ushlash.

Masalan, M13 da:

```python
try:
    pages = extract_pages(raw)
except InvalidPdfError as exc:      # 400 — fayl noto'g'ri
    ...
except ScannedPdfError as exc:       # 422 — OCR yo'q, qayta urib bo'lmaydi
    ...
except PdfAnalyzerError as exc:      # qolgan barchasi — 500
    ...
```

**Status kodlar bu yerdagi klasslarga yozilmaydi** — HTTP bilish API
qatlamining ishi. Shu sabab bu modul `fastapi` bilmaydi.
"""

__all__ = [
    "InvalidPdfError",
    "NotEnoughContentError",
    "PdfAnalyzerError",
    "ScannedPdfError",
]


class PdfAnalyzerError(Exception):
    """Loyihaning barcha domain xatolarining ildizi.

    Pipeline ichidagi kutilmagan holatlar uchun **ishlatilmaydi** — u
    `ProgrammingError` yoki `RuntimeError` uchun mo'ljallangan. Bu
    klass faqat foydalanuvchi kiritmasi bilan bog'liq nosozliklar uchun.
    """


class InvalidPdfError(PdfAnalyzerError):
    """Fayl PDF sifatida qabul qilinmadi (I/O contract §1.2: I1, I2, I5, I6, I7).

    Qamrab oladigan holatlar:
    - fayl yo'q yoki o'qilmaydi
    - bo'sh fayl (0 bayt)
    - magic bytes `%PDF` emas (masalan `.pdf` nomli PNG)
    - parol bilan himoyalangan (`needs_pass`)
    - buzilgan PDF (`pymupdf` ochmaydi)
    - kamida bitta sahifa yo'q
    """


class ScannedPdfError(PdfAnalyzerError):
    """PDF ochildi, lekin matn qatlami yo'q — skanerlangan hujjat (I4).

    OCR **yo'q** (non-goal), shuning uchun bu holatni tuzatish yo'li
    yo'q: foydalanuvchiga aniq xato beriladi va ML pipeline ishga
    tushirilmaydi.
    """


class NotEnoughContentError(PdfAnalyzerError):
    """Matn bor, lekin ML uchun yetarli emas (I/O contract §4, M4/M12).

    Skanerlangan PDF'dan **farqli**: matn qatlami mavjud, muammo
    hajmda. Masalan 40 belgi — bu o'qilgan PDF, lekin chunk'lash va
    clustering uchun ma'nosiz.
    """