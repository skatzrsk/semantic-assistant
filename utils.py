# utils.py
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def load_excel(file):
    df = pd.read_excel(file)
    df.columns = ['phrase', 'topics']
    return df

def semantic_search(query, df, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    phrase_embeddings = model.encode(df['phrase'].tolist(), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, phrase_embeddings)[0]
    top_results = similarities.argsort(descending=True)[:top_k]
    results = []

    for idx in top_results:
        phrase = df.iloc[int(idx)]['phrase']
        topics = df.iloc[int(idx)]['topics']
        results.append(f"**{phrase}** → {topics}")

    return results
