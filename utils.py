import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def load_excel(file):
    df = pd.read_excel(file)
    df = df.dropna()
    df.columns = ['phrase', 'topics']
    df['phrase'] = df['phrase'].str.strip().str.lower()
    df['embedding'] = df['phrase'].apply(lambda x: model.encode(x))
    return df

def search_similar(df, query, top_k=3, threshold=0.6):
    query = query.strip().lower()
    query_emb = model.encode(query)
    df['score'] = df['embedding'].apply(lambda x: cosine_similarity([x], [query_emb])[0][0])
    filtered = df[df['score'] >= threshold].sort_values(by='score', ascending=False)
    return filtered[['phrase', 'topics', 'score']].head(top_k)
