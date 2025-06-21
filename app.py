import streamlit as st
from utils import load_excel, search_similar

st.set_page_config(page_title="AI Разметка", layout="centered")

st.title("\ud83d\udcca AI помощник для разметки данных")
st.markdown("Загрузите Excel с фразами и тематикой. Введите свой запрос — получите результат по смыслу.")

uploaded_file = st.file_uploader("Загрузите Excel (.xlsx)", type="xlsx")

if uploaded_file:
    with st.spinner("Обрабатываем файл..."):
        df = load_excel(uploaded_file)
    st.success("Файл загружен! Введите запрос ниже.")

    query = st.text_input("Введите запрос")

    if query:
        results = search_similar(df, query)
        if not results.empty:
            st.markdown("### \ud83d\udd0d Результаты поиска:")
            for _, row in results.iterrows():
                st.markdown(f"**Фраза:** {row['phrase']}  \n**Тематики:** {row['topics']}  \n_Сходство: {row['score']:.2f}_")
        else:
            st.warning("Ничего не найдено. Попробуйте иначе сформулировать.")
