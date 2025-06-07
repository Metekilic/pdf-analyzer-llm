import os
import csv
import json
import io
from datetime import datetime, timedelta
import streamlit as st
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import (
    clean_output,
    fix_encoding,
    init_language_tool,
    summarize_documents,
    extract_key_concepts,
)


def load_css(path: str) -> None:
    """Load a local CSS file into the Streamlit app."""
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Ortam değişkeni
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.utcnow()
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "notes" not in st.session_state:
    st.session_state.notes = {}
if "tags" not in st.session_state:
    st.session_state.tags = {}

# Yazım denetimi başlat
_turk_tool = init_language_tool()

st.set_page_config(
    page_title="PDF Analyzer Chatbot",
    page_icon="📚",
    layout="centered",
)

theme = st.sidebar.selectbox("Tema", ["Aydınlık", "Karanlık"])
if theme == "Karanlık":
    load_css("dark.css")
else:
    load_css("style.css")


def load_uploaded_pdf(uploaded, db_path):
    """Load and index the uploaded PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=150
    )
    docs = text_splitter.split_documents(raw_docs)

    for doc in docs:
        doc.page_content = fix_encoding(doc.page_content)

    embed = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
    )

    vectordb = FAISS.from_documents(docs, embed)
    vectordb.save_local(db_path)
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    os.remove(pdf_path)
    return retriever, docs, raw_docs


def create_qa_chain(retriever, model_name: str):
    """Build conversational retrieval chain."""
    llm = Ollama(
        model=model_name,
        base_url="http://host.docker.internal:11434",
        temperature=0.3,
    )

    answer_prompt = PromptTemplate.from_template(
        """Aşağıdaki belgeye dayanarak soruyu yanıtla.

Yanıt verirken:
- Öncelikle belgeyi dikkatlice analiz et.
- Belgedeki doğrudan bilgileri açıkça belirt.
- Eğer doğrudan bilgi yoksa, sadece belgeye dayalı mantıklı çıkarımlar yap.
- Tahmin veya dış bilgi kullanma.
- Aynı başlığı veya maddeyi tekrar etme.
- Özetlemen isteniyorsa başlıkları sadece tekrarlama, içerikleri anlamlı şekilde sadeleştir.
- Teknik terimleri ve sayısal değerleri aynen aktar.
- Türkçe karakterleri eksiksiz koru.
- <think> gibi düşünce yazıları kullanma.

### Belge:
{context}

### Soru:
{question}

### Yanıt:
"""
    )

    condense_prompt = PromptTemplate.from_template(
        "Sadece soruyu sadeleştir: {question}"
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, k=5
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
    )

# Başlık
st.title("📚 Türkçe PDF RAG Chatbot + Qwen3:8B")
st.markdown("Qwen3:8B • Ollama • LangChain • FAISS • Çıkarım + Karakter Onarımı + Tekrar Engelleme")

db_path = "faiss_index"
uploaded = st.file_uploader("PDF yükle (.pdf)", type="pdf")
tags_input = st.text_input("Etiketler (virgül ile)")
if st.session_state.tags:
    tag_filter = st.sidebar.selectbox(
        "Etikete göre belge ara", sorted({t for tags in st.session_state.tags.values() for t in tags})
    )
    st.sidebar.write(", ".join([name for name, tags in st.session_state.tags.items() if tag_filter in tags]))

# Oturum geçmişi
if "history" not in st.session_state:
    st.session_state.history = []

# Otomatik oturum temizleme
if datetime.utcnow() - st.session_state.start_time > timedelta(hours=6):
    st.session_state.history = []
    st.session_state.pop("summary", None)
    st.session_state.start_time = datetime.utcnow()

# Sidebar actions
if st.sidebar.button("🧹 Sohbeti Temizle"):
    st.session_state.history = []
    st.session_state.pop("summary", None)

if st.session_state.history:
    chat_text = "\n".join(f"{r}: {m}" for r, m in st.session_state.history)
    st.sidebar.download_button("💾 Sohbeti İndir", chat_text, "chat_history.txt")
    csv_buf = io.StringIO()
    csv_writer = csv.writer(csv_buf)
    csv_writer.writerow(["role", "message"])
    csv_writer.writerows(st.session_state.history)
    st.sidebar.download_button(
        "⬇️ CSV", csv_buf.getvalue(), "chat_history.csv", mime="text/csv"
    )
    st.sidebar.download_button(
        "⬇️ JSON", json.dumps(st.session_state.history), "chat_history.json"
    )

if st.session_state.notes:
    notes_text = json.dumps(st.session_state.notes, indent=2)
    st.sidebar.download_button("📥 Notları İndir", notes_text, "notes.json")

# PDF yüklendiğinde çalış
if uploaded and "retriever" not in st.session_state:
    with st.spinner("PDF işleniyor..."):
        retriever, docs, pages = load_uploaded_pdf(uploaded, db_path)
        st.session_state.retriever = retriever
        st.session_state.docs = docs
        st.session_state.pages = pages
        if tags_input:
            st.session_state.tags[uploaded.name] = [t.strip() for t in tags_input.split(',') if t.strip()]

if "docs" in st.session_state:
    page_num = st.sidebar.number_input(
        "Sayfa", min_value=1, max_value=len(st.session_state.pages), step=1, value=1
    )
    st.sidebar.markdown("### Sayfa Önizleme")
    st.sidebar.write(st.session_state.pages[page_num - 1].page_content[:400] + "...")
    note_key = f"page_{page_num}_note"
    note_text = st.sidebar.text_area("Not", st.session_state.notes.get(note_key, ""))
    if st.sidebar.button("Notu Kaydet", key=f"save_{page_num}"):
        st.session_state.notes[note_key] = note_text
    st.sidebar.markdown("### Anahtar Kavramlar")
    concepts = extract_key_concepts(st.session_state.pages[page_num - 1].page_content)
    st.sidebar.write(", ".join(concepts))
    if st.sidebar.button("📰 PDF'yi Özetle"):
        with st.spinner("Özet çıkarılıyor..."):
            st.session_state.summary = summarize_documents(st.session_state.docs)
    if st.session_state.get("summary"):
        st.sidebar.markdown("### Özet")
        st.sidebar.write(st.session_state.summary)

    words = sum(len(p.page_content.split()) for p in st.session_state.pages)
    summary_len = len(st.session_state.get("summary", "").split())
    st.sidebar.markdown("### İstatistikler")
    st.sidebar.write(f"Toplam kelime: {words}")
    st.sidebar.write(f"Soru sayısı: {st.session_state.question_count}")
    st.sidebar.write(f"Özet uzunluğu: {summary_len}")

# Model ve zincir
if "retriever" in st.session_state:
    model_name = st.sidebar.selectbox(
        "Model", ["qwen3:8b", "mistral", "phi3"], index=0
    )
    qa = create_qa_chain(st.session_state.retriever, model_name)

    q = st.chat_input("Sorunuz")
    if q:
        with st.spinner("Cevap hazırlanıyor..."):
            try:
                result = qa.invoke({"question": q})
                raw_answer = result["answer"]
            except Exception as e:
                st.error(f"Model çağrısında hata oluştu: {e}")
                raw_answer = "**YETERSİZ VERİ**"

        response = clean_output(raw_answer)

        if _turk_tool and 10 < len(response) < 200:
            try:
                response = _turk_tool.correct(response)
            except Exception:
                pass

        st.session_state.question_count += 1
        st.session_state.history += [("Siz", q), ("Asistan", response)]

    for role, msg in st.session_state.history:
        with st.chat_message("user" if role == "Siz" else "assistant"):
            st.markdown(msg)

else:
    st.info("Lütfen bir PDF yükleyin.")

st.sidebar.markdown(f"**Model:** {model_name}")
#docs = st.session_state.retriever.get_relevant_documents(q)
#context = "\n".join([doc.page_content for doc in docs])
#st.markdown("### 🔍 Modelin Gördüğü Bağlam:")
#st.write(context)

