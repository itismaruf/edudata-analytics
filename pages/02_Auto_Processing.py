import streamlit as st
import pandas as pd
import time
from Utils.automatic_data_processing import run_auto_cleaning
from Utils.outlier_utils import run_auto_outlier_removal



st.title("🛡️ Автоматический обработка")

with st.expander("🧭 Как пользоваться этим разделом"):
    st.write(
        "Нажмите кнопку ниже, чтобы автоматически очистить данные. "
        "Будет выполнена универсальная обработка: сначала удаляются или корректируются пропуски, "
        "а затем выявляются и устраняются выбросы. "
        "В результате вы получите очищенный датасет, готовый к дальнейшему анализу."
    )


if "df" not in st.session_state:
    st.warning("Сначала загрузите данные в разделе 📥 'Загрузка данных'.", icon="⚠️")
else:
    df = st.session_state["df"]

    if st.button("🫧 Автообработка данных"):
        # Шаг 1: автоочистка пропусков
        with st.spinner("Шаг 1/2: обработка пропусков…"):
            time.sleep(2)  # имитация процесса
            try:
                stats_before, clean_log, df = run_auto_cleaning(df)
                st.session_state["auto_clean_log"] = clean_log
            except Exception as e:
                st.error(f"Ошибка при обработке пропусков: {e}", icon="🚫")

        # Шаг 2: автообработка выбросов
        with st.spinner("Шаг 2/2: обработка выбросов…"):
            time.sleep(2)  # имитация процесса
            try:
                before_df, outlier_log, df = run_auto_outlier_removal(df)
                st.session_state["auto_outlier_log"] = outlier_log
            except Exception as e:
                st.error(f"Ошибка при обработке выбросов: {e}", icon="🚫")

        # Сохраняем результат
        st.session_state["df"] = df
        st.session_state["auto_proc_done"] = True

    # --- Отображение результатов (Persistence) ---
    if st.session_state.get("auto_proc_done"):
        clean_log = st.session_state.get("auto_clean_log", [])
        outlier_log = st.session_state.get("auto_outlier_log", [])

        if len(clean_log) == 0:
            st.info("ℹ️ Пропусков не найдено — очистка не потребовалась.")
        else:
            st.success("✅ Пропуски успешно обработаны")
            st.write(pd.DataFrame(clean_log))
        
        if len(outlier_log) == 0:
            st.info("ℹ️ Выбросов не обнаружено — обработка не потребовалась.")
        else:
            st.success("✅ Выбросы успешно обработаны")
