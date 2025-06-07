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
- 🧹 Sohbeti tek tıkla temizleme
- 📝 Sohbet geçmişini indirme
- 📰 PDF'yi özetleme
- 🎨 Göz yormayan modern arayüz

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
```

Uygulama kodu `main.py` dosyasında başlayıp yardımcı işlevlerin yer aldığı
`utils.py` dosyasını kullanır. PDF yükleme ve model zinciri oluşturma gibi
işlemler bu yardımcı modüller aracılığıyla yürütülür.

---

## ✨ Yeni Arayüz ve Özellikler

Uygulama Streamlit bileşenleri ve özel CSS kullanılarak modern bir görünüme
kavuşturuldu. Aşağıda önceki ve yeni arayüzden kesitler yer almaktadır:

```
Önceki görünüm:
-------------------------------------
| 📚 Türkçe PDF RAG Chatbot + Qwen3 |
| PDF yükle alanı                  |
| Sorunuz: _______________________ |
-------------------------------------

Yeni görünüm (özet):
-------------------------------------
| 📚 PDF Analyzer Chatbot          |
| PDF yükle alanı                  |
| Chat baloncukları ve alt kısımda |
| mesaj girişi                    |
-------------------------------------
```

Yeni tasarım sade renkler, daha büyük butonlar ve mobil uyumlu bir yerleşim
sunar. Chat mesajları baloncuk şeklinde gösterilir ve tüm denetimler tek tıkla
erişilebilir.

### Eklenen Başlıca Özellikler

- Sayfa bazlı önizleme ve not bırakma
- Karanlık/Aydınlık tema seçimi
- PDF için etiket ekleme ve etiket ile arama
- İstatistik paneli ve sohbet geçmişini CSV/JSON olarak indirme
- Model seçici ile farklı Ollama modellerini kullanabilme
- Rol tabanlı oturum açma ve çoklu kullanıcı yönetimi
- Kullanıcı işlemlerini `audit.log` dosyasında tutan denetim kaydı
- Oturum ve yükleme sayıları için basit analitik toplama
- GitHub Actions ve Kubernetes tanımları ile konteynerleşmiş dağıtım
