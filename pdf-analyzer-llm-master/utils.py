# Utility functions for the PDF chatbot
import re
import language_tool_python


def init_language_tool():
    """Initialize Turkish language tool if available."""
    try:
        return language_tool_python.LanguageTool("tr")
    except Exception:
        return None


def fix_encoding(text: str) -> str:
    """Fix common encoding issues in PDF text."""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return (
        text.replace("ý", "ı")
        .replace("þ", "ş")
        .replace("ð", "ğ")
        .replace("Ý", "İ")
        .replace("Þ", "Ş")
        .replace("Ð", "Ğ")
    )


def clean_output(text: str) -> str:
    """Remove extraneous tags and whitespace."""
    text = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.I | re.S)
    text = re.sub(r"<.*?>", "", text, flags=re.I | re.S)
    text = re.sub(r"\s{2,}", " ", text)
    return fix_encoding(text.strip())
