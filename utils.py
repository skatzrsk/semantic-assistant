# utils.py
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]


def load_excel(url):
    df = pd.read_excel(url)
    df = df[['phrase', 'topics1', 'topics2', 'topics3', 'topics4', 'topics5', 'topics6']]
    df['topics'] = df[['topics1', 'topics2', 'topics3', 'topics4', 'topics5', 'topics6']].astype(str).agg(', '.join, axis=1)
    return df[['phrase', 'topics']]


def load_all_excels():
    dfs = [load_excel(url) for url in GITHUB_CSV_URLS]
    return pd.concat(dfs, ignore_index=True)


def semantic_search(query, df):
    vectorizer = TfidfVectorizer().fit(df['phrase'].tolist() + [query])
    query_vec = vectorizer.transform([query])
    phrase_vecs = vectorizer.transform(df['phrase'])
    sims = cosine_similarity(query_vec, phrase_vecs).flatten()
    top_indices = sims.argsort()[-5:][::-1]  # Top 5
    results = [(df.iloc[i]['phrase'], df.iloc[i]['topics'].split(', ')) for i in top_indices if sims[i] > 0.3]
    return results
