"""Ichki ma'lumot modellari — `docs/io_contract.md` §2.1.

Bu modellar **tashqi API bilan bog'liq emas**: FastAPI ularni
`response_model` sifatida ishlatmaydi, chunki javob namunasi boshqa
(`docs/io_contract.md` §3). Ular pipeline ichidagi typed ma'lumotni
ta'minlaydi — `str` va `dict` aralashib ketmasligi uchun.
"""

from pydantic import BaseModel, Field

__all__ = ["PageText"]


class PageText(BaseModel):
    """Bitta sahifaning xom matni.

    Qoidalar (`docs/io_contract.md` §2.1):
    - `page` — **1-based**, foydalanuvchi PDF viewer'da ko'radigan raqam.
    - Bo'sh sahifalar **o'chirilmaydi** — statistika uchun kerak
      (muqova, ajratgich, rasm).
    - `text` — `.strip()` qilingan.

    Attributes:
        page: Sahifa raqami, 1 dan boshlanadi.
        text: Xom matn. Bo'sh sahifada `""`.
    """

    page: int = Field(ge=1, description="1-based sahifa raqami")
    text: str = Field(description="Xom matn (.strip qilingan)")

    @property
    def is_empty(self) -> bool:
        """Sahifada o'qiladigan matn yo'qmi."""
        return not self.text.strip()

    @property
    def n_chars(self) -> int:
        """Matn uzunligi — hisoblashda `len(text)` ni qayta yozmaslik uchun."""
        return len(self.text)