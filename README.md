# 🎬 IMDb Film Öneri Asistanı – RAG Tabanlı Chatbot

Bu proje, **IMDb film veri seti** üzerinde çalışan, **RAG (Retrieval-Augmented Generation)** mimarisiyle desteklenmiş bir film öneri chatbot sistemidir. Kullanıcılar doğal dilde film tercihlerini ifade edebilir; sistem bu tercihlere en uygun filmleri gerçek IMDb verilerinden çıkararak akıllı öneriler sunar.

🔗 **Canlı Demo (Deploy Linki):** [https://imdb-film-oneri-asistani.streamlit.app/](https://imdb-film-oneri-asistani.streamlit.app/)  
*(Lütfen kendi Streamlit deploy linkinizi buraya güncelleyin.)*

---

## 📌 Projenin Amacı

Kullanıcıların zevklerine, tür tercihlerine, puan aralıklarına veya yıllara göre kişiselleştirilmiş film önerileri alabilmelerini sağlamak. Sistem, sadece statik filtreleme yerine, **anlamsal benzerlik** ve **doğal dil anlama** yeteneğiyle dinamik ve bağlamsal öneriler sunar.

---

## 🗃️ Veri Seti Hakkında

- **Kaynak:** [Hugging Face – jquigl/imdb-genres](https://huggingface.co/datasets/jquigl/imdb-genres)  
  (Orijinal veri: [Kaggle – IMDb Movies by Genre](https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre))
- **İçerik:** Toplam **297.821 film** (train + test + validation)
- **Kolonlar:**
  - `movie title - year`: Film adı ve yılı (örn. `"Inception - 2010"`)
  - `genre`: Ana tür (örn. `"Sci-Fi"`)
  - `expanded-genres`: Tüm türler (virgülle ayrılmış)
  - `rating`: IMDb puanı (1–10 arası)
  - `description`: Kısa özet (eksik açıklamalar temizlenmiş)

> Proje, performans nedeniyle varsayılan olarak **1.000 film** ile başlar. Kullanıcı sidebar’dan 100–5.000 arası örnek boyutunu seçebilir. Veri seti eksikse, 25 popüler film içeren **demo modu** devreye girer.

---

## ⚙️ Kullanılan Yöntemler

### 🧠 RAG (Retrieval-Augmented Generation) Mimarisi

1. **Veri Hazırlama:**  
   Hugging Face’ten alınan jquigl/imdb-genres veri seti işlenerek her film için aşağıdaki formatta bir içerik metni oluşturulur:
    - Title: Inception
    - Genre: Sci-Fi
    - Rating: 8.8/10
    - Year: 2010
    - Description: A thief who steals corporate secrets through dream-sharing...  

2. **Embedding (Alıntı):**  
- Tüm filmler ve kullanıcı sorgusu **Google Gemini `text-embedding-004`** modeliyle vektörleştirilir.
- Vektörler, bellek içi (in-memory) bir NumPy array’de saklanır (küçük veri seti olduğu için FAISS/Chroma gibi harici vektör veritabanına gerek duyulmamıştır).
- Kullanıcı sorgusu geldiğinde, aynı modelle sorgu da vektörleştirilir.

3. **Retrieval (Getirme):**  
- Kullanıcı sorgusunun embedding’i ile tüm film embedding’leri arasında cosine similarity hesaplanır.
- En yüksek benzerlik skoruna sahip 5 film seçilir.
- Bu filmler, LLM’e bağlam (context) olarak iletilir.

4. **Generation (Üretme):**  
- Seçilen filmler ve kullanıcı sorgusu, Gemini gemini-2.0-flash-exp modeline özel bir prompt ile iletilir.
- Prompt, modelin:
    - Sadece sağlanan filmlerden öneri yapmasını,
    - IMDb puanını, türünü ve açıklamasını içermesini,
    - Kullanıcının diline (Türkçe/İngilizce) göre yanıt vermesini sağlar.
- Model, doğal, akıcı ve bilgilendirici bir öneri metni üretir.

### 🛠️ Kullanılan Teknolojiler
| Bileşen | Teknoloji | Açıklama
|--------|----------| ---------- |
| LLM & Embedding | Google Gemini API | text-embedding-004
(embedding), gemini-2.0-flash-exp (generation) |
| Veri Seti | Hugging Face (`jquigl/imdb-genres`) | 298K film içeren, tür, puan ve açıklama bilgisiyle zenginleştirilmiş veri |
| Web Arayüzü | Streamlit | Gerçek zamanlı chat arayüzü, sidebar ile yapılandırılabilir parametreler |
| Ortam Yönetimi | python-dotenv | API anahtarının güvenli yönetimi |
|RAG Framework | Özelleştirilmiş pipeline (LangChain/Haystack kullanılmadı)| Basit, şeffaf ve hafif bir mimari tercih edildi|
| Vektör Depolama | NumPy (in-memory) | Performans ve basitlik için harici vektör DB’si kullanılmadı|
---

## 📊 Elde Edilen Sonuçlar

- Sistem, **doğal dildeki karmaşık sorgulara** (örn. *“8 üzeri psikolojik gerilim filmleri”*) anlamlı yanıtlar verebiliyor.
- Gerçek IMDb verileri sayesinde öneriler **güvenilir ve güncel**.
- Demo modu sayesinde **API veya veri seti erişimi olmayan kullanıcılar** bile sistemi test edebiliyor.
- Arayüz, kullanıcı deneyimini artırmak için **örnek sorular**, **veri istatistikleri** ve **interaktif chat** içeriyor.

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.9+
- Google Gemini API Key ([Nasıl alınır?](https://ai.google.dev/))
- requirements.txt
    - haystack-ai
    - google-genai-haystack
    - datasets
    - pandas
    - sentence-transformers
    - python-dotenv
    - streamlit
    - numpy

### Adımlar
1. Projeyi klonlayın:
```bash
git clone https://github.com/nsaylan/IMDB_Film_Oneri_Asistani.git
cd IMDB_Film_Oneri_Asistani
```
2. Sanal ortam oluşturun ve bağımlılıkları yükleyin:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. .env dosyası oluşturun ve API anahtarınızı ekleyin:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```
4. Uygulamayı başlatın:
```bash
streamlit run app.py
```
💡 Not: API anahtarı olmadan uygulama demo modunda (25 popüler filmle) çalışır. 

## 🌐 Web Arayüzü & Ürün Kılavuzu
- **Arayüz**: Streamlit ile geliştirilmiştir.
- ### Kullanım Akışı:
```bash
1. Sayfa yüklendiğinde sistem otomatik olarak veri setini ve vektör veritabanını hazırlar.
2. Kullanıcı sidebar’dan veri boyutunu seçebilir.
3. Chat kutusuna film tercihini doğal dilde yazar (örn. “Komedi filmleri öner”).
4. Sistem, RAG pipeline’ı üzerinden anında öneri sunar.
```
- ### Özellikler:

    - Türkçe/İngilizce destek
    - Örnek sorularla hızlı başlangıç
    - Gerçek zamanlı veri istatistikleri
    - Hata durumunda fallback mekanizması

## 📚 Kaynaklar
- Google Gemini API Dokümantasyonu
- Hugging Face – jquigl/imdb-genres
- Akbank GenAI Bootcamp Proje Rehberi


🎯 Akbank GenAI Bootcamp Projesi – RAG Tabanlı Chatbot Geliştirme
Geliştirici: [Necip Saylan]
Tarih: Ekim 2025 
