# 🎬 IMDb Film Öneri Asistanı – RAG Tabanlı Chatbot

Bu proje, **IMDb film veri seti** üzerinde çalışan, **RAG (Retrieval-Augmented Generation)** mimarisiyle desteklenmiş bir film öneri chatbot sistemidir. Kullanıcılar doğal dilde film tercihlerini ifade edebilir; sistem bu tercihlere en uygun filmleri gerçek IMDb verilerinden çıkararak akıllı öneriler sunar.


---

## 📌 Projenin Amacı

Kullanıcıların zevklerine, tür tercihlerine, puan aralıklarına veya yıllara göre kişiselleştirilmiş film önerileri alabilmelerini sağlamak.

---

## 🗃️ Veri Seti Hakkında

- **Veri Kaynağı:** Hugging Face Datasets
- **Dataset:** [jquigl/imdb-genres](https://huggingface.co/datasets/jquigl/imdb-genres)
- **İçerik:** Orijinal veri seti yaklaşık **298K** film kaydı içermektedir.
- **Kullanım Metodolojisi:** Performans ve Streamlit'in ücretsiz katman kısıtlamaları göz önünde bulundurularak, uygulamanın anlık gereksinimine göre (örneğin 1000, 5000) veri setinin yalnızca ilk $N$ filmi yüklenip işlenmektedir.
- **Kullanılan Sütunlar:**
  - `movie title - year`: Film adı ve yılı (örn. `"Inception - 2010"`)
  - `genre`: Ana tür (örn. `"Sci-Fi"`)
  - `expanded-genres`: Tüm türler (virgülle ayrılmış)
  - `rating`: IMDb puanı (1–10 arası)
  - `description`: Kısa özet (eksik açıklamalar temizlenmiş)
- **Veri Temizleme:** Başlık, yıl, tür, puan ve açıklama gibi temel alanlar ayıklanmış, temizlenmiş ve RAG için kullanılacak tek bir content alanında birleştirilmiştir.

> Proje, performans nedeniyle varsayılan olarak **1.000 film** ile başlar. Kullanıcı sidebar’dan 100–5.000 arası örnek boyutunu seçebilir. Veri seti eksikse, 25 popüler film içeren **demo modu** devreye girer.

---

## ⚙️ Kullanılan Yöntemler ve Çözüm Mimarisi (RAG)

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
- Kullanıcı sorgusunun embedding’i ile tüm film embedding’leri arasında **Cosine Similarity** hesaplanır.
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
| LLM & Embedding | Google Gemini API | text-embedding-004 (embedding), gemini-2.0-flash-exp (generation) |
| Veri Seti | Hugging Face (`jquigl/imdb-genres`) | 298K film içeren, tür, puan ve açıklama bilgisiyle zenginleştirilmiş veri |
| Web Arayüzü | Streamlit | Gerçek zamanlı chat arayüzü, sidebar ile yapılandırılabilir parametreler |
| Ortam Yönetimi | python-dotenv | API anahtarının güvenli yönetimi |
|RAG Framework | Haystack | Karmaşık RAG Pipeline'larının kolayca oluşturulmasını ve yönetimini sağlayan bir framework|
| Vektör Depolama | NumPy (in-memory) | Performans ve basitlik için harici vektör DB’si kullanılmadı|
| Veri İşleme | Pandas, NumPy, Hugging Face Datasets|	Film verilerini yükleme, temizleme ve işleme|
---

## 📊 Elde Edilen Sonuçlar

- Sistem, **doğal dildeki karmaşık sorgulara** (örn. *“8 üzeri psikolojik gerilim filmleri”*) anlamlı yanıtlar verebiliyor.
- Gerçek IMDb verileri sayesinde öneriler **güvenilir ve güncel**.
- Demo modu sayesinde **API veya veri seti erişimi olmayan kullanıcılar** bile sistemi test edebiliyor.
- Streamlit sayesinde uygulama, **kullanıcı dostu** bir arayüze sahiptir. Arayüz, kullanıcı deneyimini artırmak için **örnek sorular**, **veri istatistikleri** ve **interaktif chat** içeriyor. 
- Chatbot, kullanıcıların diline **(Türkçe/İngilizce)** uyum sağlayarak, film önerilerini kullanıcı tarafından belirtilen türe, puana veya yıla göre filtreleyip sunabilmektedir.

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
```
3. Projenin bağımlılıklarını requirements.txt dosyasından yükleyin:
```bash
pip install -r requirements.txt
```
4. .env dosyası oluşturun ve API anahtarınızı ekleyin:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```
5. Uygulamayı başlatın:
```bash
streamlit run app.py
```
💡 Not: API anahtarı olmadan uygulama demo modunda (25 popüler filmle) çalışır. 

## 🌐 Web Arayüzü & Ürün Kılavuzu
🔗 **Canlı Demo (Deploy Linki):** [https://imdb-film-oneri-asistani.streamlit.app/]
- **Arayüz**: Streamlit ile geliştirilmiştir.
- ### Kullanım Akışı:
```bash
1. Sayfa yüklendiğinde sistem otomatik olarak veri setini ve vektör veritabanını hazırlar. 
2. Kullanıcı sidebar’dan veri boyutunu seçebilir. Sidebar'da seçilen Veri Seti Boyutuna göre Hugging Face'den veri seti yüklenir ve tüm filmler vektörleştirilerek in-memory veritabanı oluşturulur. Bu aşamalar, yükleme süresi boyunca spinner ile belirtilir.
3. Kullanıcı, chat kutusuna film tercihini yazar (örn. “Komedi filmleri öner”).
4. Sistem, RAG pipeline’ı üzerinden:
  - Sorgu vektörleştirilir.
  - En benzer 5 film bilgisi geri çekilir.
  - Bu 5 film bilgisi, Gemini 2.0 Flash'a gönderilerek akıcı ve bilgilendirici bir öneri metni üretilir.
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
