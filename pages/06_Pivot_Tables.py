import streamlit as st
import pandas as pd
from Utils.visualization import show_pivot_tab



st.title("📟 Сводные таблицы (Pivot)")
st.caption("ℹ В этом разделе вы можете строить сводные таблицы и визуализировать их.")

if "df" not in st.session_state:
    st.warning("📥 Сначала загрузите данные.", icon="⚠️")
else:
    df = st.session_state["df"]
    show_pivot_tab(df)
