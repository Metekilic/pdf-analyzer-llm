# 📄 PDF Analyzer LLM Chatbot

Türkçe destekli, FAISS tabanlı, Streamlit arayüzlü bir PDF analiz chatbot uygulaması.  
Belirttiğiniz PDF dosyaları üzerinden veri çıkarımı yapar, semantik arama ve doğal dil ile soru-cevap özellikleri sunar.

---

## 🚀 Özellikler

- 🧠 **LLM (Large Language Model)** destekli PDF anlama
- 🔎 **FAISS vektör veritabanı** ile hızlı içerik arama
- 📁 Çoklu PDF dosya desteği
- 💬 Streamlit tabanlı kullanıcı dostu arayüz
- 🔌 Ollama destekli yerel model çalıştırma (örneğin: `qwen3:8b`, `mistral`, `phi3`)

---

## 🧰 Gereksinimler

- Docker & Docker Compose
- Ollama (https://ollama.com)
- Qwen3:8b Açık kaynaklı LLM modeli
- Python 3.10+ (yalnızca geliştirme ortamı için)

---

## 🧱 Kurulum (Docker ile)

```bash
git clone https://github.com/Metekilic/pdf-analyzer-llm.git
cd pdf-analyzer-llm

# Ollama'da bir model başlat
ollama run qwen3:8b

# Uygulamayı başlat
docker-compose up --build
