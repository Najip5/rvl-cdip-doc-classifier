"""
OCR Preprocessor Module for PT Mobil Sejuta Umat / RVL-CDIP
Menyediakan standarisasi pra-pemrosesan teks OCR bising (noisy OCR)
dan pipeline ColumnTransformer yang dapat diserialisasi utuh.
"""

import re
import string
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def clean_noisy_ocr_text(text: str) -> str:
    """
    Membersihkan teks pindaian OCR bising:
    - Menghapus karakter kontrol & karakter non-ASCII aneh (scan artifacts)
    - Menormalkan spasi berlebih, baris baru, dan tabulasi
    - Menormalkan tanda baca
    - Mengubah ke huruf kecil (lowercase)
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Ganti non-ASCII / artifact noise dengan spasi
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # 2. Hapus karakter noise kontrol / bintik scan umum (e.g. ~, _, ^, \, |, `)
    cleaned = re.sub(r'[\~_\^\\\|`]', ' ', cleaned)
    
    # 3. Normalkan spasi berulang
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned.lower()

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer untuk membersihkan teks OCR
    secara otomatis di dalam Pipeline sebelum proses TfidfVectorizer.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.fillna("").apply(clean_noisy_ocr_text)
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].fillna("").apply(clean_noisy_ocr_text)
        elif isinstance(X, list):
            return [clean_noisy_ocr_text(str(x)) for x in X]
        else:
            return pd.Series(X).fillna("").apply(clean_noisy_ocr_text)
