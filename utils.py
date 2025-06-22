# utils.py
import pandas as pd
import requests
from io import BytesIO
from sentence_transformers import SentenceTransformer, util

# Semantic model initialization
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Ссылки на Excel-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download {url} — status {response.status_code}")
    df = pd.read_excel(BytesIO(response.content))
    if 'phrase' not in df.columns:
        raise KeyError(f"'phrase' column missing in {url}")
    
    # Динамический список всех topics-колонок
    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError(f"No 'topics*' columns found in {url}")
    
    # Оставляем только phrase + найденные topics
    df = df[['phrase'] + topic_cols]
    # Объединяем темы в один список
    df['topics'] = df[topic_cols].fillna('').agg(lambda x: [t for t in x.tolist() if t], axis=1)
    return df[['phrase', 'topics']]

def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            df = load_excel(url)
            dfs.append(df)
        except Exception as e:
            st = ""  # Здесь можно логировать, если нужно
    if not dfs:
        raise ValueError("No valid Excel files found")
    return pd.concat(dfs, ignore_index=True)

def semantic_search(query, df, top_k=5, threshold=0.5):
    query_emb = model.encode(query, convert_to_tensor=True)
    phrase_embs = model.encode(df['phrase'].tolist(), convert_to_tensor=True)
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

