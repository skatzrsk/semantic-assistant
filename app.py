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
