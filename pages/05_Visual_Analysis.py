import streamlit as st
import pandas as pd
from Utils.visualization import show_chart_tab, show_ai_suggestions, show_correlation_tab



st.title("📊 Визуальный анализ")
st.caption('ℹ В этом разделе вы можете сделать визуальный анализ!')

if "df" not in st.session_state:
    st.warning("📥 Сначала загрузите данные.", icon="⚠️")
else:
    df = st.session_state["df"]
    tabs = st.tabs(["📊 Графики", "📈 Корреляции"])

    with tabs[0]:
        show_ai_suggestions(df)
        show_chart_tab(df)  

    with tabs[1]:
        show_correlation_tab(df)
