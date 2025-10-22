# -*- coding: utf-8 -*-
"""
IMDb Film Recommendation Chatbot - RAG System
============================================

Bu proje, IMDb film veri setini kullanarak RAG (Retrieval Augmented Generation) 
tabanlı bir film öneri chatbot sistemi oluşturur.

Kullanılan Teknolojiler:
- Google Gemini API: Doğal dil üretimi ve embedding
- Streamlit: Web arayüzü
- Pandas: Veri işleme
- NumPy: Vektör işlemleri
- Hugging Face Datasets: jquigl/imdb-genres veri seti

Veri Seti: 298K film içeren IMDb genres dataset
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from dotenv import load_dotenv
from datasets import load_dataset

# Ortam değişkenlerini yükle
load_dotenv()
dataset = load_dataset("jquigl/imdb-genres")

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="IMDb Film Öneri Asistanı",
    page_icon="🎬",
    layout="wide"
)

# --- 1. Adım: API Anahtarını Yükleme ---
def load_environment():
    """Ortam değişkenlerini yükle ve API anahtarını kontrol et."""
    try:
        # Environment'tan API key al
        google_api_key = os.getenv("GEMINI_API_KEY")
        
        # Streamlit secrets'tan dene
        if not google_api_key:
            try:
                google_api_key = st.secrets.get("GEMINI_API_KEY")
            except:
                pass
        
        if not google_api_key:
            st.error("⚠️ GEMINI_API_KEY bulunamadı.")
            st.info("""
            💡 **API Key Nasıl Alınır:**
            1. https://ai.google.dev/ adresini ziyaret edin
            2. Google hesabınızla giriş yapın
            3. 'Get API Key' butonuna tıklayın
            4. Oluşturduğunuz API Key'i `.env` dosyasına ekleyin
            
            **📝 .env Dosyası Formatı:**
            ```
            GEMINI_API_KEY=your_api_key_here
            ```
            """)
            st.stop()
            
        return google_api_key
    except Exception as e:
        st.error(f"❌ Ortam değişkenleri yüklenirken hata oluştu: {e}")
        st.stop()

# --- 2. Adım: Hugging Face IMDb Veri Setini Yükleme ---
@st.cache_data(show_spinner=False)
def load_imdb_data_from_hf(sample_size=1000):
    """
    Hugging Face'den IMDb film veri setini yükler.
    
    Args:
        sample_size: Yüklenecek film sayısı (performans için sınırlı)
        
    Returns:
        pd.DataFrame: İşlenmiş film verileri
    """
    try:
        with st.spinner(f"🎬 Hugging Face'den IMDb veri seti yükleniyor... (İlk {sample_size} film)"):
            # Hugging Face'den veri setini yükle
            dataset = load_dataset("jquigl/imdb-genres", split="train", trust_remote_code=True)
            
            st.info(f"📊 Toplam {len(dataset)} film bulundu. İlk {sample_size} film kullanılacak.")
            
            # İlk N kaydı al
            dataset = dataset.select(range(min(sample_size, len(dataset))))
            
            # DataFrame'e çevir
            df = pd.DataFrame(dataset)
            
            # Veri seti yapısını kontrol et
            available_cols = df.columns.tolist()
            
            with st.expander("🔍 Veri Seti Bilgileri"):
                st.write("**Mevcut Kolonlar:**", available_cols)
                st.write("**İlk Satır Örneği:**")
                st.json(df.iloc[0].to_dict())
            
            # jquigl/imdb-genres dataset kolon yapısı:
            # "movie title - year", "genre", "expanded-genres", "rating", "description"
            
            # Kolon isimlendirmesini düzelt
            if 'movie title - year' in df.columns:
                # Title ve year'ı ayır
                df[['title', 'year_str']] = df['movie title - year'].str.extract(r'(.+?)\s*-\s*(\d{4})')
                df['year'] = pd.to_numeric(df['year_str'], errors='coerce').fillna(2000).astype(int)
                df = df.drop(columns=['movie title - year', 'year_str'])
            else:
                st.warning("⚠️ 'movie title - year' kolonu bulunamadı, alternatif yapı kullanılıyor...")
                # Alternatif kolon isimleri için arama
                for col in available_cols:
                    col_lower = col.lower()
                    if 'title' in col_lower and 'title' not in df.columns:
                        df['title'] = df[col]
                    elif 'year' in col_lower and 'year' not in df.columns:
                        df['year'] = pd.to_numeric(df[col], errors='coerce').fillna(2000).astype(int)
            
            # Genre kontrolü
            if 'genre' not in df.columns:
                if 'expanded-genres' in df.columns:
                    # İlk genre'yi al
                    df['genre'] = df['expanded-genres'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')
                else:
                    df['genre'] = 'Unknown'
            
            # Rating kontrolü
            if 'rating' in df.columns:
                df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(7.0).astype(float)
            else:
                df['rating'] = 7.0
            
            # Description kontrolü
            if 'description' not in df.columns:
                for col in ['plot', 'summary', 'overview']:
                    if col in df.columns:
                        df['description'] = df[col]
                        break
                if 'description' not in df.columns:
                    df['description'] = "No description available"
            
            # Varsayılan değerler
            if 'title' not in df.columns:
                st.error("❌ Film başlıkları bulunamadı!")
                st.stop()
            if 'year' not in df.columns:
                df['year'] = 2000
            
            # Sadece gerekli kolonları tut
            df = df[['title', 'year', 'genre', 'rating', 'description']].copy()
            
            # Veri temizleme
            df = df.dropna(subset=['title', 'description'])
            df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2000).astype(int)
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(7.0).astype(float)
            df['genre'] = df['genre'].fillna('Unknown').astype(str)
            
            # Çok uzun açıklamaları kısalt (embedding için)
            df['description'] = df['description'].apply(lambda x: str(x)[:500] if len(str(x)) > 500 else str(x))
            
            # Film içeriklerini birleştir (RAG için)
            df['content'] = df.apply(lambda row: f"""Title: {row['title']}
Genre: {row['genre']}
Rating: {row['rating']}/10
Year: {row['year']}
Description: {row['description']}""", axis=1)
            
            st.success(f"✅ {len(df)} film başarıyla yüklendi!")
            
            # Örnek veri göster
            with st.expander("📊 Örnek Film Verileri (İlk 5 Film)"):
                st.dataframe(df[['title', 'year', 'genre', 'rating']].head())
            
            return df
            
    except Exception as e:
        st.error(f"❌ Hugging Face veri seti yüklenirken hata oluştu: {e}")
        st.warning("🔄 Alternatif olarak demo veri seti kullanılıyor...")
        
        # FALLBACK: Demo veri seti
        sample_movies = [
            {"title": "The Shawshank Redemption", "year": 1994, "genre": "Drama", "rating": 9.3,
             "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."},
            {"title": "The Godfather", "year": 1972, "genre": "Crime", "rating": 9.2,
             "description": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
            {"title": "The Dark Knight", "year": 2008, "genre": "Action", "rating": 9.0,
             "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological tests."},
            {"title": "Pulp Fiction", "year": 1994, "genre": "Crime", "rating": 8.9,
             "description": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption."},
            {"title": "Forrest Gump", "year": 1994, "genre": "Drama", "rating": 8.8,
             "description": "The presidencies of Kennedy and Johnson, the Vietnam War, and other historical events unfold from the perspective of an Alabama man."},
            {"title": "Inception", "year": 2010, "genre": "Sci-Fi", "rating": 8.8,
             "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the task of planting an idea."},
            {"title": "The Matrix", "year": 1999, "genre": "Sci-Fi", "rating": 8.7,
             "description": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."},
            {"title": "Interstellar", "year": 2014, "genre": "Sci-Fi", "rating": 8.6,
             "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},
            {"title": "Gladiator", "year": 2000, "genre": "Action", "rating": 8.5,
             "description": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family."},
            {"title": "The Prestige", "year": 2006, "genre": "Mystery", "rating": 8.5,
             "description": "After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion."},
            {"title": "The Lion King", "year": 1994, "genre": "Animation", "rating": 8.5,
             "description": "Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself."},
            {"title": "Parasite", "year": 2019, "genre": "Thriller", "rating": 8.6,
             "description": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."},
            {"title": "Whiplash", "year": 2014, "genre": "Drama", "rating": 8.5,
             "description": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor."},
            {"title": "Joker", "year": 2019, "genre": "Drama", "rating": 8.4,
             "description": "In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society."},
            {"title": "Avengers: Endgame", "year": 2019, "genre": "Action", "rating": 8.4,
             "description": "After the devastating events of Infinity War, the Avengers assemble once more to reverse Thanos' actions."},
            {"title": "Titanic", "year": 1997, "genre": "Romance", "rating": 7.9,
             "description": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."},
            {"title": "The Notebook", "year": 2004, "genre": "Romance", "rating": 7.8,
             "description": "A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom."},
            {"title": "Toy Story", "year": 1995, "genre": "Animation", "rating": 8.3,
             "description": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy."},
            {"title": "The Hangover", "year": 2009, "genre": "Comedy", "rating": 7.7,
             "description": "Three buddies wake up from a bachelor party in Las Vegas, with no memory of the previous night."},
            {"title": "Superbad", "year": 2007, "genre": "Comedy", "rating": 7.6,
             "description": "Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to stage a party goes awry."},
            {"title": "The Conjuring", "year": 2013, "genre": "Horror", "rating": 7.5,
             "description": "Paranormal investigators work to help a family terrorized by a dark presence in their farmhouse."},
            {"title": "A Quiet Place", "year": 2018, "genre": "Horror", "rating": 7.5,
             "description": "In a post-apocalyptic world, a family is forced to live in silence while hiding from monsters with ultra-sensitive hearing."},
            {"title": "Spider-Man: Into the Spider-Verse", "year": 2018, "genre": "Animation", "rating": 8.4,
             "description": "Teen Miles Morales becomes the Spider-Man of his universe and must join with five spider-powered individuals."},
            {"title": "The Social Network", "year": 2010, "genre": "Biography", "rating": 7.7,
             "description": "As Harvard student Mark Zuckerberg creates the social networking site that would become Facebook."},
            {"title": "Eternal Sunshine of the Spotless Mind", "year": 2004, "genre": "Romance", "rating": 8.3,
             "description": "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories."},
        ]
        
        df = pd.DataFrame(sample_movies)
        
        # Sample size'a göre sınırla
        if len(df) > sample_size:
            df = df.head(sample_size)
        
        st.info(f"📝 Demo modu: {len(df)} popüler IMDb filmi kullanılıyor.")
        
        # Film içeriklerini birleştir
        df['content'] = df.apply(lambda row: f"""Title: {row['title']}
Genre: {row['genre']}
Rating: {row['rating']}/10
Year: {row['year']}
Description: {row['description']}""", axis=1)
        
        return df

# --- 3. Adım: Embedding Oluşturma ---
def create_embeddings(texts, client, batch_size=50):
    """
    Metinler için Gemini embedding'leri oluştur.
    
    Args:
        texts: List of strings
        client: Gemini client
        batch_size: Batch boyutu
        
    Returns:
        numpy array: Embedding vektörleri
    """
    embeddings = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        status_text.text(f"🔄 Embedding oluşturuluyor: Batch {batch_num}/{total_batches}")
        
        try:
            result = client.models.embed_content(
                model="models/text-embedding-004",
                contents=batch
            )
            for emb in result.embeddings:
                embeddings.append(emb.values)
        except Exception as e:
            st.warning(f"⚠️ Embedding hatası (batch {batch_num}): {e}")
            # Hata durumunda rastgele vektör ekle (fallback)
            for _ in batch:
                embeddings.append(np.random.randn(768).tolist())
        
        # Progress bar güncelle
        progress_bar.progress(min((i + batch_size) / len(texts), 1.0))
    
    progress_bar.empty()
    status_text.empty()
    return np.array(embeddings)

@st.cache_resource(show_spinner=False)
def create_vector_store(_df, _client):
    """
    Film verileri için vektör deposu oluştur.
    
    Args:
        _df: Film DataFrame
        _client: Gemini client
        
    Returns:
        tuple: (DataFrame, embeddings array)
    """
    with st.spinner("🔍 Film vektör veritabanı oluşturuluyor..."):
        try:
            texts = _df['content'].tolist()
            embeddings = create_embeddings(texts, _client)
            st.success(f"✅ {len(embeddings)} film için embedding oluşturuldu!")
            return _df, embeddings
        except Exception as e:
            st.error(f"❌ Vektör veritabanı oluşturulurken hata: {e}")
            return _df, None

# --- 4. Adım: Similarity Search ---
def search_similar_movies(query, client, df, embeddings, top_k=5):
    """
    Sorguya en benzer filmleri bul.
    
    Args:
        query: Kullanıcı sorgusu
        client: Gemini client
        df: Film DataFrame
        embeddings: Film embeddings
        top_k: Döndürülecek film sayısı
        
    Returns:
        pd.DataFrame: En benzer filmler
    """
    try:
        # Query için embedding oluştur
        query_result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=[query]
        )
        query_embedding = np.array(query_result.embeddings[0].values)
        
        # Cosine similarity hesapla
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # En yüksek benzerlik skorlarına sahip indeksleri al
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Similarity skorlarını da ekle
        result_df = df.iloc[top_indices].copy()
        result_df['similarity'] = similarities[top_indices]
        
        return result_df
        
    except Exception as e:
        st.error(f"❌ Arama hatası: {e}")
        # Hata durumunda rastgele filmler dön
        return df.sample(min(top_k, len(df)))

# --- 5. Adım: RAG Pipeline ---
def generate_recommendation(query, client, df, embeddings):
    """
    RAG kullanarak film önerisi üret.
    
    Args:
        query: Kullanıcı sorgusu
        client: Gemini client
        df: Film DataFrame
        embeddings: Film embeddings
        
    Returns:
        str: Öneri metni
    """
    try:
        # Benzer filmleri bul
        similar_movies = search_similar_movies(query, client, df, embeddings, top_k=5)
        
        # Film bilgilerini context'e ekle
        movies_context = ""
        for idx, row in similar_movies.iterrows():
            movies_context += f"""
Movie: {row['title']}
Genre: {row['genre']}
Rating: {row['rating']}/10
Year: {row['year']}
Description: {row['description']}
---
"""
        
        # Prompt oluştur
        prompt = f"""You are a professional movie recommendation assistant. Based on the provided movie information, 
give personalized movie recommendations that match the user's request.

Instructions:
- Provide specific movie recommendations from the given movies
- Include movie titles, genres, ratings, and brief descriptions
- If the user asks about a specific genre, year, or rating range, focus on those criteria
- Keep recommendations concise but informative
- Always mention the IMDb rating when recommending
- Format your response in a friendly, conversational way
- Respond in the same language as the user's question (Turkish if they ask in Turkish, English if they ask in English)

Available Movies:
{movies_context}

User Request: {query}

Movie Recommendations:"""
        
        # Gemini ile yanıt oluştur
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        return response.text if response.text else "Üzgünüm, şu anda film önerisi üretemedim."
        
    except Exception as e:
        return f"❌ Film önerisi üretilirken hata: {str(e)}"

# --- 6. Adım: Streamlit Arayüzü ---
def main():
    """Ana Streamlit uygulaması"""
    
    # Başlık
    st.title("🎬 IMDb Film Öneri Asistanı")
    st.caption("RAG teknolojisi ile desteklenen akıllı film öneri sistemi")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Sistem Bilgileri")
        
        # Veri seti boyutu seçimi
        sample_size = st.selectbox(
            "Veri Seti Boyutu",
            options=[100, 500, 1000, 2000, 5000],
            index=2,
            help="Daha büyük veri seti daha uzun yükleme süresi anlamına gelir"
        )
        
        st.info(f"""
        **Veri Kaynağı:** Hugging Face  
        **Dataset:** jquigl/imdb-genres  
        **Yüklenecek Film:** {sample_size}  
        **Embedding Model:** text-embedding-004  
        **Generation Model:** gemini-2.0-flash-exp
        """)
        
        st.header("💡 Örnek Sorular")
        examples = [
            "En iyi aksiyon filmleri nelerdir?",
            "8+ puana sahip drama filmleri öner",
            "2010 sonrası bilim kurgu filmleri",
            "Komedi filmleri öner",
            "Romantik filmler arasında favorilerin neler?",
            "2000'lerin en iyi filmleri",
            "Thriller türünde filmler"
        ]
        
        for ex in examples:
            if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
                st.session_state.example_query = ex
        
        # İstatistikler
        if 'df' in st.session_state and st.session_state.df is not None:
            st.header("📈 Veri İstatistikleri")
            df_stats = st.session_state.df
            st.metric("Toplam Film", len(df_stats))
            st.metric("Ortalama Rating", f"{df_stats['rating'].mean():.1f}")
            st.metric("Yıl Aralığı", f"{df_stats['year'].min()}-{df_stats['year'].max()}")
    
    # API anahtarını yükle
    api_key = load_environment()
    
    # Gemini client oluştur
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Gemini client oluşturulamadı: {e}")
        st.stop()
    
    # Veri yükleme (cache key olarak sample_size kullan)
    cache_key = f"data_{sample_size}"
    if 'cache_key' not in st.session_state or st.session_state.cache_key != cache_key:
        st.session_state.cache_key = cache_key
        st.session_state.df = None
        st.session_state.embeddings = None
    
    if st.session_state.df is None:
        df = load_imdb_data_from_hf(sample_size=sample_size)
        if df is None or len(df) == 0:
            st.error("❌ Film veri seti yüklenemedi.")
            st.stop()
        st.session_state.df = df
    else:
        df = st.session_state.df
    
    # Vektör veritabanı oluştur
    if st.session_state.embeddings is None:
        df_indexed, embeddings = create_vector_store(df, client)
        if embeddings is None:
            st.error("❌ Vektör veritabanı oluşturulamadı.")
            st.stop()
        st.session_state.embeddings = embeddings
    else:
        df_indexed = df
        embeddings = st.session_state.embeddings
    
    st.success(f"✅ Sistem hazır! {len(df)} film yüklendi.")
    
    # Chat geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Hoş geldin mesajı
    if not st.session_state.messages:
        welcome = f"""👋 **Merhaba! IMDb Film Öneri Asistanınıza hoş geldiniz.**

Sistemde **{len(df)} gerçek IMDb filmi** bulunuyor. Size nasıl film önerileri verebilirim?

**Örnek sorular:**
- "Romantik filmler öner."
- "8+ puana sahip drama filmleri öner"
- "2010 sonrası bilim kurgu filmleri"

Sormak istediğiniz soruyu aşağıdaki chat kutusuna yazabilirsiniz! 🎭"""
        st.session_state.messages.append({"role": "assistant", "content": welcome})
    
    # Chat geçmişini göster
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Örnek sorguları işle
    if hasattr(st.session_state, 'example_query'):
        user_input = st.session_state.example_query
        del st.session_state.example_query
    else:
        user_input = st.chat_input("Film önerisi için sorunuzu yazın...")
    
    # Kullanıcı girdisini işle
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # RAG ile film önerisi
        with st.chat_message("assistant"):
            with st.spinner("🔍 Film veritabanında arama yapıyor..."):
                response = generate_recommendation(user_input, client, df_indexed, embeddings)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
    🎬 IMDb Film Öneri Asistanı | RAG + Gemini AI | Akbank GenAI Bootcamp Projesi<br>
    📊 Veri Kaynağı: <a href="https://huggingface.co/datasets/jquigl/imdb-genres" target="_blank">Hugging Face - jquigl/imdb-genres</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()