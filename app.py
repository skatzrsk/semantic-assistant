├── semantic-search-app/
│   ├── app.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── README.md
│   └── .gitignore

# app.py
import streamlit as st
from utils import load_all_excels, semantic_search

st.set_page_config(page_title="Semantic Assistant", layout="wide")
st.title("🤖 Semantic Assistant for Labeled Phrases")

query = st.text_input("Введите ваш запрос")

if query:
    df = load_all_excels()
    results = semantic_search(query, df)
    st.markdown("### 🔍 Результаты поиска:")
    for phrase, topics in results:
        st.markdown(f"- **{phrase}** → {', '.join(topics)}")

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

# requirements.txt
streamlit
pandas
scikit-learn
openpyxl

# README.md
# Semantic Search Assistant

AI-powered assistant that semantically matches user queries to labeled phrases and returns associated topics.

## Features
- Takes user query and matches it to labeled phrases from multiple Excel files
- Each phrase can have up to 6 associated topics
- Data files are stored on GitHub and loaded remotely

## How to Run
1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/semantic-search-app.git
cd semantic-search-app
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the app
```bash
streamlit run app.py
```

## Excel File Format
Each Excel file (e.g. data1.xlsx) should have:
- A column `phrase`
- Columns `topics1` through `topics6`

# .gitignore
__pycache__/
*.pyc
.env
.DS_Store
