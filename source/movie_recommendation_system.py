import pandas as pd
import numpy as np
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from google.colab import drive
drive.mount('/content/drive')



movies_path = "/content/drive/MyDrive/ML Proje/Movie Recommendation System/tmdb_5000_movies.csv"
credits_path = "/content/drive/MyDrive/ML Proje/Movie Recommendation System/tmdb_5000_credits.csv"

"""**Veri setlerini okuma**"""

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

print("Movies shape:", movies.shape)
print("Credits shape:", credits.shape)

"""**Veri setlerini birleştirme**"""

movies = movies.merge(credits, on="title")
movies.shape


"""
**Gerekli sütunları seçme**
"""

movies = movies[[
    "movie_id",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew"
]]



movies.head()

"""**Eksik Verileri Kontrol Etme**"""

movies.isnull().sum()

"""**Eksik verileri güvenli şekilde temizleme**"""

movies.dropna(subset=[
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew"
], inplace=True)



movies.isnull().sum()
print("After dropna shape:", movies.shape)

"""**genres ve keywords sütunlarını parse etme**"""

def convert(text):
    result = []
    for item in ast.literal_eval(text):
        result.append(item["name"])
    return result

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)


def get_top_cast(text):
    result = []
    counter = 0

    for item in ast.literal_eval(text):
        if counter < 3:
            result.append(item["name"])
            counter += 1
        else:
            break

    return result

"""Şimdi Uygulayalım:"""

movies["cast"] = movies["cast"].apply(get_top_cast)

"""
**crew sütunundan yönetmeni alma**
"""

def fetch_director(text):
    result = []

    for item in ast.literal_eval(text):
        if item["job"] == "Director":
            result.append(item["name"])
            break

    return result

"""Şimdi Uygulayalım:"""

movies["crew"] = movies["crew"].apply(fetch_director)

"""
**overview sütununu kelime listesine çevirme**
"""

movies["overview"] = movies["overview"].apply(lambda x: x.split())

"""
**Çok Kelimeli İsimlerde Boşluk Kaldırma**
"""

movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

"""
**tags sütunu oluşturma**
"""

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

"""Kontrol:"""

movies[["title", "tags"]].head()

"""
**Yeni dataframe oluşturma**
"""

new_df = movies[["movie_id", "title", "tags"]].copy()

"""
**tags listesini düz metne çevirme**
"""

new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

"""Kontrol:"""

new_df.head()

"""
**Metinleri sayısal vektöre dönüştürme**
"""

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

"""Kontrol:"""

print("Vectors shape:", vectors.shape)

"""
**Cosine similarity hesaplama**
"""

similarity = cosine_similarity(vectors)

"""Kontrol:"""

print("Similarity matrix shape:", similarity.shape)

"""
***Öneri Fonksiyonu Yazma***
"""

def recommend(movie):
    movie = movie.lower()

    matching_movies = new_df[new_df["title"].str.lower() == movie]

    if matching_movies.empty:
        print("Film bulunamadı.")
        return

    index = matching_movies.index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    print(f"\n'{matching_movies.iloc[0]['title']}' için önerilen filmler:\n")

    for i in movie_list[1:6]:
        print(new_df.iloc[i[0]].title)

"""
# **Sistemi Test Etme**
"""

recommend("Avatar")

recommend("Inception")

recommend("The Dark Knight")



