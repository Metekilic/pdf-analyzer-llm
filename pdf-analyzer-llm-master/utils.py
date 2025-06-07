# Utility functions for the PDF chatbot
import re
import language_tool_python
import os
import json
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama


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


def summarize_documents(docs, max_chunks: int = 10) -> str:
    """Generate a brief summary from the provided documents."""
    text = "\n".join(doc.page_content for doc in docs[:max_chunks])
    llm = Ollama(
        model="qwen3:8b",
        base_url="http://host.docker.internal:11434",
        temperature=0.3,
    )
    prompt = PromptTemplate.from_template(
        "Aşağıdaki belge içeriğini kısaca özetle:\n{context}\n\nÖzet:"
    )
    summary = llm.invoke(prompt.format(context=text))
    return clean_output(summary)


def extract_key_concepts(text: str, n: int = 5) -> list[str]:
    """Return a list of key concept words."""
    words = re.findall(r"\b\w{5,}\b", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]]


def log_event(user: str, action: str) -> None:
    """Append an audit log entry to audit.log."""
    line = f"{datetime.utcnow().isoformat()}," f"{user},{action}\n"
    with open("audit.log", "a", encoding="utf-8") as f:
        f.write(line)


def record_user_stat(user: str, key: str) -> None:
    """Increment a usage counter for the given user and key."""
    path = "analytics.json"
    stats = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception:
            stats = {}
    stats.setdefault(user, {})
    stats[user][key] = stats[user].get(key, 0) + 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f)

