import streamlit as st
import pandas as pd
from Utils.upload_utils import load_data, get_base_info, show_data_head, show_descriptive_stats, display_base_info
from Utils.AI_helper import connect_ai_context

st.title("📥 Загрузка данных")
st.caption("Загрузите ваш файл в формате CSV или Excel для начала анализа.")

# --- Загрузка данных ---
if "df" not in st.session_state:
    uploaded_file = st.file_uploader(" ", type=["csv", "xlsx", "xls"])
    if not uploaded_file:
        st.info("⬆ Загрузите файл для анализа.", icon="📁")
    else:
        try:
            df = load_data(uploaded_file)
            st.session_state["df"] = df
            st.success("Данные успешно загружены", icon="✅")
        except Exception as e:
            st.error(f"Ошибка при обработке данных: {e}", icon="🚫")
else:
    df = st.session_state["df"]
    st.success("Данные уже загружены ✅")

# --- Если данные загружены ---
if "df" in st.session_state:
    st.markdown("---")

    # Превью данных в экспандере
    with st.expander("Пример данных (первые строки)", expanded=False):
        show_data_head(df)

    # Описательная статистика в отдельном экспандере
    with st.expander("📑 Описательная статистика", expanded=False):
        show_descriptive_stats(df)

    # Метрики
    base_info = get_base_info(df)
    display_base_info(base_info)

    # — Инициализация/обновление краткого summary —
    data_sig = (tuple(df.columns), df.shape)
    if st.session_state.get("_data_sig") != data_sig:
        summary = f"{df.shape[0]} строк, {df.shape[1]} столбцов; признаки: {', '.join(map(str, df.columns))}"
        st.session_state["_data_sig"] = data_sig
        st.session_state["data_summary"] = summary
        
        # update_context removed as state is now unified in chat history
    else:
        summary = st.session_state.get(
            "data_summary",
            f"{df.shape[0]} строк, {df.shape[1]} столбцов; признаки: {', '.join(map(str, df.columns))}"
        )

    st.markdown("---")
    # Блок подключения ИИ в экспандере
    st.subheader("🤖 Подключение ИИ")
    st.caption("Нажмите кнопку ниже, чтобы передать контекст данных ИИ. Это позволит ему отвечать на вопросы по вашему датасету.")

    if st.button("Подключить ИИ"):
        with st.spinner("Подключаем ИИ к вашим данным..."):
            import time
            time.sleep(2)
            connect_ai_context(df)
        st.success("✅ ИИ успешно подключен к данным! Теперь вы можете переходить в чат.")
