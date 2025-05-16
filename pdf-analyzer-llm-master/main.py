import os
import streamlit as st
import tempfile
import re
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import language_tool_python

# Ortam değişkeni
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Yazım denetimi başlat
try:
    _turk_tool = language_tool_python.LanguageTool("tr")
except Exception:
    _turk_tool = None

# Karakter düzeltici
def fix_encoding(text: str) -> str:
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = text.replace("ý", "ı").replace("þ", "ş").replace("ð", "ğ") \
               .replace("Ý", "İ").replace("Þ", "Ş").replace("Ð", "Ğ")
    return text

# Yanıt temizleyici
def clean_output(text: str) -> str:
    text = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.I | re.S)
    text = re.sub(r"<.*?>", "", text, flags=re.I | re.S)
    text = re.sub(r"\s{2,}", " ", text)
    return fix_encoding(text.strip())

# Başlık
st.title("📚 Türkçe PDF RAG Chatbot + Qwen3:8B")
st.markdown("Qwen3:8B • Ollama • LangChain • FAISS • Çıkarım + Karakter Onarımı + Tekrar Engelleme")

db_path = "faiss_index"
uploaded = st.file_uploader("PDF yükle (.pdf)", type="pdf")

# Oturum geçmişi
if "history" not in st.session_state:
    st.session_state.history = []

# PDF yüklendiğinde çalış
if uploaded and "retriever" not in st.session_state:
    with st.spinner("PDF işleniyor..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150)
        docs = text_splitter.split_documents(raw_docs)

        # PDF içeriğini önceden karakter düzelt
        for doc in docs:
            doc.page_content = fix_encoding(doc.page_content)

        embed = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"}
        )

        vectordb = FAISS.from_documents(docs, embed)
        vectordb.save_local(db_path)
        #retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        retriever = vectordb.as_retriever(search_kwargs={"k": 8})

        st.session_state.retriever = retriever
        st.session_state.docs = docs

        os.remove(pdf_path)

# Model ve zincir
if "retriever" in st.session_state:
    llm = Ollama(
        model="qwen3:8b",
        base_url="http://host.docker.internal:11434",
        temperature=0.3
    )

    answer_prompt = PromptTemplate.from_template("""
Aşağıdaki belgeye dayanarak soruyu yanıtla.

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
""")

    condense_prompt = PromptTemplate.from_template("Sadece soruyu sadeleştir: {question}")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=5)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.retriever,
        memory=memory,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt}
    )

    q = st.text_input("Sorunuz:")
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

        st.session_state.history += [("Siz", q), ("Asistan", response)]

    for role, msg in st.session_state.history:
        st.markdown(f"**{role}:** {msg}")

else:
    st.info("Lütfen bir PDF yükleyin.")

st.sidebar.markdown("**Model:** qwen3:8b - Ollama GGUF")
#docs = st.session_state.retriever.get_relevant_documents(q)
#context = "\n".join([doc.page_content for doc in docs])
#st.markdown("### 🔍 Modelin Gördüğü Bağlam:")
#st.write(context)

