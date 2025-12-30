import streamlit as st
import pandas as pd
import io
import os
import time
from Utils.outlier_utils import (detect_outliers_iqr, detect_outliers_zscore, plot_outliers_distribution, outliers_summary, run_auto_outlier_removal, render_outlier_rules_table, remove_outliers_iqr, remove_outliers_zscore, cap_outliers, remove_outliers_percentile, plot_outlier_removal_comparison)



st.title("🚩 Обработка выбросов")

# Инициализация флага изменений
if "data_changed" not in st.session_state:
    st.session_state["data_changed"] = False

if "df" not in st.session_state:
    st.warning("📥 Загрузите данные на предыдущей странице", icon="⚠️")
else:
    df = st.session_state["df"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Анализ и визуализация выбросов
    st.subheader("🔍 Анализ выбросов")
    with st.expander("👁 Просмотреть распределение выбросов"):
        cols_viz = st.multiselect(
            "Выберите числовые столбцы для анализа",
            numeric_cols,
            key="out_viz_cols"
        )
        method_viz = st.radio(
            "Метод обнаружения выбросов",
            ["IQR-метод", "Z-score"],
            key="out_viz_method"
        )

        if method_viz == "IQR-метод":
            q_low, q_high = st.slider(
                "Квантили для IQR",
                0.0, 0.5, (0.25, 0.75),
                step=0.05,
                key="iqr_viz"
            )
        else:
            z_thresh = st.number_input(
                "Порог Z-score",
                min_value=1.0, max_value=5.0,
                value=3.0, step=0.1,
                key="z_viz"
            )

        if st.button("👁 Показать выбросы", key="show_out_viz"):
            masks = (detect_outliers_iqr(df, cols_viz, q_low, q_high)
                        if method_viz == "IQR-метод"
                        else detect_outliers_zscore(df, cols_viz, z_thresh))
            fig = plot_outliers_distribution(df, masks, cols_viz)
            st.plotly_chart(fig, use_container_width=True)

            summary = outliers_summary(df, masks)
            st.table(summary.set_index("column"))

    st.markdown("---")

    # Автоочистка выбросов
    st.subheader("🤖 Автообработка выбросов")
    with st.expander("📌 Правила автообработки выбросов"):
        render_outlier_rules_table()

    if st.button("🚀 Запустить автоочистку выбросов", key="auto_out"):
        with st.spinner("⏳ Обработка выбросов..."):
            time.sleep(1.0)
            before, log, cleaned_df = run_auto_outlier_removal(df)
            st.session_state["df"] = cleaned_df
            st.session_state["outlier_auto_log"] = log
            st.session_state["df_before_outlier"] = before
            st.session_state["outlier_auto_done"] = True
        
    # --- Отображение результатов автоочистки (Persistence) ---
    if st.session_state.get("outlier_auto_done"):
        log = st.session_state.get("outlier_auto_log", [])
        cleaned_df = st.session_state["df"]
        
        total_removed = sum(item["removed_count"] for item in log)
        if total_removed == 0:
            st.info("Автоматически выбросы не найдены", icon="✅")
        else:
            report = (
                pd.DataFrame(log)
                    .rename(columns={
                        "column": "Столбец",
                        "method": "Метод",
                        "removed_count": "Удалено выбросов"
                    })
                    .set_index("Столбец")
            )
            st.markdown("**Отчет автоочистки выбросов**")
            st.table(report)
            st.success(f"Удалено выбросов: {total_removed}")

            st.markdown("### Сравнение распределений до и после автоочистки")
            fig_cmp = plot_outlier_removal_comparison(st.session_state.get("df_before_outlier", df), cleaned_df, numeric_cols)
            st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")

    st.subheader("🔧 Ручная очистка выбросов")
    with st.expander("✍️ Панель ручной очистки выбросов", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cols_manual = st.multiselect(
                "Столбцы для обработки",
                numeric_cols,
                key="out_manual_cols"
            )
        with col2:
            method_manual = st.selectbox(
                "Метод обработки",
                [
                    "Удалить выбросы (IQR)",
                    "Каппинг (IQR-границы)",
                    "Удаление по Z-score",
                    "Удаление по процентилям"
                ],
                key="out_manual_method"
            )

        # Параметры для каждого метода
        if method_manual in ("Удалить выбросы (IQR)", "Каппинг (IQR-границы)"):
            low_q, high_q = st.slider(
                "Квантили для IQR",
                0.0, 0.5, (0.25, 0.75),
                step=0.05,
                key="iqr_manual"
            )
        elif method_manual == "Удаление по Z-score":
            z_manual = st.number_input(
                "Порог Z-score",
                min_value=1.0, max_value=5.0,
                value=3.0, step=0.1,
                key="z_manual"
            )
        else:  # Удаление по процентилям
            p_low, p_high = st.slider(
                "Процентили для удаления",
                0, 100, (5, 95),
                step=1,
                key="percentile_manual"
            )

        if st.button("✅ Применить ручную очистку"):
            before_manual = df.copy()
            cleaned_manual = df.copy()

            for col in cols_manual:
                if method_manual == "Удалить выбросы (IQR)":
                    cleaned_manual = remove_outliers_iqr(cleaned_manual, [col], low_q, high_q)
                elif method_manual == "Каппинг (IQR-границы)":
                    cleaned_manual = cap_outliers(cleaned_manual, [col], low_q, high_q)
                elif method_manual == "Удаление по Z-score":
                    cleaned_manual = remove_outliers_zscore(cleaned_manual, [col], z_manual)
                else:
                    cleaned_manual = remove_outliers_percentile(cleaned_manual, [col], p_low, p_high)

            st.session_state["df"] = cleaned_manual
            st.success("✅ Ручная очистка выбросов завершена")

            # show_outlier_summary(before_manual, cleaned_manual, cols_manual)

            st.markdown("### Сравнение распределений до и после ручной очистки")
            fig_cmp_manual = plot_outlier_removal_comparison(
                before_manual, cleaned_manual, cols_manual
            )
            st.plotly_chart(fig_cmp_manual, use_container_width=True)

    # === 📥 Кнопка скачивания, если были изменения ===
    if st.session_state.get("data_changed", False) and not st.session_state["df"].empty:
        st.markdown("---")
        st.subheader("📥 Скачать обработанные данные")

        base_name = "data"
        if "original_filename" in st.session_state:
            base_name = os.path.splitext(st.session_state["original_filename"])[0]
        file_name = f"{base_name}_cleaned.csv"

        csv_buffer = io.BytesIO()
        st.session_state["df"].to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.success("✅ Файл готов к скачиванию")
        st.download_button(
            label=f"💾 Скачать {file_name}",
            data=csv_buffer,
            file_name=file_name,
            mime="text/csv"
        )
