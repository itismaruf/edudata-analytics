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

saved_results = st.session_state.get("stats_test_results")
if saved_results:
    with st.expander("💾 Последний сохраненный результат", expanded=True):
        st.caption(f"Тест: {saved_results.get('test', 'не указан')}")
        metric_cols = st.columns(2)
        metric_cols[0].metric(
            saved_results.get("stat_label", "Статистика"),
            f"{saved_results.get('stat_value', 0):.4f}",
        )
        metric_cols[1].metric("p-value", f"{saved_results.get('p_value', 0):.4f}")
        if saved_results.get("figure") is not None:
            st.plotly_chart(saved_results["figure"], use_container_width=True)
        if saved_results.get("summary") is not None:
            st.dataframe(saved_results["summary"], use_container_width=True)
        if saved_results.get("observed") is not None:
            st.subheader("Наблюдаемые значения")
            st.dataframe(saved_results["observed"], use_container_width=True)
        if saved_results.get("expected") is not None:
            st.subheader("Ожидаемые значения")
            st.dataframe(saved_results["expected"], use_container_width=True)

# --- Запуск соответствующего UI ---
if selected_test == "t-test":
    show_ttest_ui(df)      # обновлённый вариант с колонками
elif selected_test == "ANOVA":
    show_anova_ui(df)      # обновлённый вариант с колонками
else:
    show_chi2_ui(df)       # обновлённый вариант с колонками
