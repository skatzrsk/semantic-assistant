# utils.py
import pandas as pd
import requests
import re
import string
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

# Используем более мощную модель для лучшего понимания смысла
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Ссылки на Excel-файлы в GitHub
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

# Предобработка текста — чистка, нормализация
def preprocess(text):
    text = str(text).lower()
    text = text.strip()
    text = re.sub(rf"[{string.punctuation}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Загрузка одного Excel-файла
def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download {url} — status {response.status_code}")
    
    df = pd.read_excel(BytesIO(response.content))
    
    if 'phrase' not in df.columns:
        raise KeyError(f"'phrase' column missing in {url}")
    
    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError(f"No 'topics*' columns found in {url}")
    
    df = df[['phrase'] + topic_cols]
    df['topics'] = df[topic_cols].fillna('').agg(lambda x: [t for t in x.tolist() if t], axis=1)
    df['phrase_proc'] = df['phrase'].apply(preprocess)
    return df[['phrase', 'phrase_proc', 'topics']]

# Объединение всех Excel-файлов
def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            df = load_excel(url)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Ошибка при загрузке {url}: {e}")
    if not dfs:
        raise ValueError("No valid Excel files loaded")
    return pd.concat(dfs, ignore_index=True)

# Семантический поиск по загруженному DataFrame
def semantic_search(query, df, top_k=5, threshold=0.4):
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)
    
    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]
    results = []

    for idx, score in enumerate(sims):
        score = float(score)
        if score >= threshold:
            phrase = df.iloc[idx]['phrase']
            topics = df.iloc[idx]['topics']
            results.append((score, phrase, topics))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

