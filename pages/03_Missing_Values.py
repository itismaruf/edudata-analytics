import streamlit as st
import pandas as pd
import time
from Utils.automatic_data_processing import (summarize_missing, render_nan_rules_table, run_auto_cleaning, 
                                                apply_manual_cleaning, show_na_summary, prepare_csv_download )



st.title("⚙️ Обработка пропусков")
st.caption('Обработка пропущенных значений (NaN)')

# Инициализация флага изменений
if "data_changed" not in st.session_state:
    st.session_state["data_changed"] = False

if "df" not in st.session_state:
    st.warning("📥 Загрузите данные", icon="⚠️")
else:
    df = st.session_state["df"]

    # 🎯 Выбор целевой переменной
    target = st.selectbox(
        "Целевая переменная (ее NaN будут удалены)",
        [None] + list(df.columns)
    )

    st.markdown("---")

    # 📊 Статистика пропусков
    st.subheader("📊 Пропуски в данных")
    missing = summarize_missing(df)

    if missing.empty:
        st.success("Нет пропусков в данных", icon="✅")
    else:
        st.table(
            missing.rename(columns={
                "column": "Столбец",
                "missing_count": "Кол-во",
                "pct_missing": "% пропусков"
            }).set_index("Столбец")
        )

        st.markdown("---")

        # 🤖 Автоочистка
        st.subheader("🤖 Автоочистка")
        with st.expander("📌 Правила автоочистки"):
            render_nan_rules_table()

        if st.button("🚀 Запустить автоочистку"):
            with st.spinner("⏳ Обработка пропусков..."):
                time.sleep(1.0)
                before, log, new_df = run_auto_cleaning(df, target_col=target)
                st.session_state["df"] = new_df
                st.session_state["data_changed"] = True
                st.session_state["missing_auto_log"] = log
                st.session_state["missing_auto_before"] = before
                st.session_state["missing_auto_done"] = True

        # --- Отображение результатов автоочистки (Persistence) ---
        if st.session_state.get("missing_auto_done"):
            before = st.session_state.get("missing_auto_before", pd.DataFrame())
            log = st.session_state.get("missing_auto_log", [])
            new_df = st.session_state["df"]

            if before.empty:
                st.info("Пропусков не найдено", icon="✅")
            else:
                st.markdown("**До очистки**")
                st.table(
                    before.rename(columns={
                        "column": "Столбец",
                        "missing_count": "Кол-во",
                        "pct_missing": "% пропусков"
                    }).set_index("Столбец")
                )

                report = pd.DataFrame(log).rename(columns={
                    "column": "Столбец",
                    "missing_count": "Кол-во",
                    "pct_missing": "% пропусков",
                    "action": "Действие"
                }).set_index("Столбец")

                st.markdown("**Отчет автоочистки**")
                st.table(report)

                remaining = new_df.isna().sum().sum()
                st.success(f"Готово! Осталось пропусков: {remaining}")

    st.markdown("---")

    # 🔧 Ручная очистка
    st.subheader("🔧 Ручная очистка")
    with st.expander("✍️ Панель ручной очистки"):
        cols = st.multiselect(
            "Столбцы для обработки:",
            [c for c in df.columns if c != target]
        )
        action = st.radio(
            "Действие:",
            ["Удалить строки", "Удалить столбцы (с NaN)", "Заполнить NaN",
                "Удалить выбранные столбцы", "Удалить дубликаты"]
        )
        show_tables = st.checkbox("Показывать сводку по NaN", value=True)

        method = value = None
        if action == "Заполнить NaN":
            method = st.selectbox("Метод заполнения:", ["mean", "median", "mode", "constant"])
            if method == "constant":
                value = st.text_input("Значение для заполнения:")

        if st.button("✅ Применить"):
            before = df.copy()
            new_df = apply_manual_cleaning(df, action, cols, target, method, value)

            st.session_state["df"] = new_df
            st.session_state["data_changed"] = True
            st.success("✅ Обработка завершена")

            if show_tables and action != "Удалить выбранные столбцы":
                show_na_summary(before, new_df, cols)
            elif show_tables and action == "Удалить выбранные столбцы":
                st.markdown("**Размер до/после (строки, столбцы)**")
                col1, col2 = st.columns(2)
                col1.write(before.shape)
                col2.write(new_df.shape)

    # 📥 Кнопка скачивания
    if st.session_state.get("data_changed", False) and not st.session_state["df"].empty:
        st.markdown("---")
        st.subheader("📥 Скачать обработанные данные")

        file_name, csv_buffer = prepare_csv_download(
            st.session_state["df"],
            st.session_state.get("original_filename")
        )

        st.success("✅ Файл готов к скачиванию")
        st.download_button(
            label=f"💾 Скачать {file_name}",
            data=csv_buffer,
            file_name=file_name,
            mime="text/csv"
        )
