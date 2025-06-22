# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from utils import load_excel, semantic_search

st.set_page_config(page_title="Semantic Assistant", layout="centered")
st.title("AI assistant for data labeling")

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = load_excel(uploaded_file)
        st.success("File successfully loaded. You can now enter a query.")

        query = st.text_input("Enter your query")

        if query:
            results = semantic_search(query, df)

            st.markdown("### Search results:")
            if results:
                for match in results:
                    st.write(f"- {match}")
            else:
                st.info("No relevant topics found for your query.")

    except Exception as e:
        st.error(f"Failed to load file: {e}")
