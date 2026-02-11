"""
Configuration Tesseract pour chemins custom
"""
import pytesseract
from pathlib import Path

# Chemin custom Tesseract
TESSERACT_PATH = Path(r"E:\Logiciels\Programmation\Tesseract\tesseract.exe")

if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)
    print(f"✅ Tesseract configuré : {TESSERACT_PATH}")
else:
    print(f"⚠️  Tesseract introuvable à : {TESSERACT_PATH}")
    print(f"   Installation standard attendue : C:\\Program Files\\Tesseract-OCR")