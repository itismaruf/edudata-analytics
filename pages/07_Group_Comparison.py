import streamlit as st
import pandas as pd
from Utils.stats_tests import show_ttest_ui, show_anova_ui, show_chi2_ui



st.title("⚖️ Сравнение групп")
st.caption("ℹ Проверка гипотез: t‑test, ANOVA и Chi‑square")

if "df" not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
    st.warning("📥 Сначала загрузите данные.")
    st.stop()

df = st.session_state.df

# --- Подсказка по выбору теста ---
with st.expander("🧭 Как выбрать сравнение?", expanded=False):
    st.markdown("""
    - **t‑test** — 2 группы, числовая метрика → сравнение средних  
    - **ANOVA** — 3+ групп, числовая метрика → сравнение средних  
    - **Chi‑square** — 2 категориальных признака → проверка связи между категориями
    """)

# --- Выбор теста ---
st.subheader("⚙️ Настройки")
selected_test = st.selectbox(
    "Выберите тест",
    ["t-test", "ANOVA", "Chi-squared"],
    key="stats_test_choice"
)

st.markdown("---")

# --- Запуск соответствующего UI ---
if selected_test == "t-test":
    show_ttest_ui(df)      # обновлённый вариант с колонками
elif selected_test == "ANOVA":
    show_anova_ui(df)      # обновлённый вариант с колонками
else:
    show_chi2_ui(df)       # обновлённый вариант с колонками
