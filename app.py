# ============= Модули
import streamlit as st
import pandas as pd
import os
import time
import io
from sklearn.model_selection import train_test_split
import numpy as np
import numpy

from Utils.upload_utils import load_data, get_base_info, show_data_head, show_descriptive_stats, display_base_info

from Utils.automatic_data_processing import (summarize_missing, render_nan_rules_table, run_auto_cleaning, \
                                                apply_manual_cleaning, show_na_summary, prepare_csv_download )

from Utils.outlier_utils import (detect_outliers_iqr, detect_outliers_zscore, \
    plot_outliers_distribution, outliers_summary, run_auto_outlier_removal, render_outlier_rules_table, \
    remove_outliers_iqr, remove_outliers_zscore, cap_outliers, remove_outliers_percentile, plot_outlier_removal_comparison)

from Utils.visualization import show_chart_tab, show_ai_suggestions, show_correlation_tab, show_pivot_tab

from Utils.stats_tests import show_ttest_ui, show_anova_ui, show_chi2_ui

from Utils.modeling_utils import ensure_modeling_state, sticky_selectbox, show_model_settings, \
                                 prepare_features_and_target, train_logistic_regression, evaluate_model, \
                                 compute_feature_importance, interpret_feature_importance, mark_model_trained, \
                                 show_results_and_analysis, show_single_prediction, show_export_buttons

from Utils.catboost_modeling import (
    detect_task,
    prepare_features_and_target_catboost,
    train_catboost_universal,
    evaluate_catboost_universal,
    compute_catboost_feature_importance,
    plot_feature_importance_signed,
    build_confusion_matrix,
    valid_eval_metrics_for_task,
    predict_new_object
)

from Utils.chat import continue_chat, render_message, reset_chat_history

from AI_helper import update_context, get_chatgpt_response, notify_ai_dataset_and_goal

from Utils.ui_config import (
    setup_page,
    show_splash,
    init_api_key,
    init_session,
    init_page_state,
    setup_sidebar
)

def set_page(page_name):
    st.session_state['page'] = page_name

# --- Запуск конфигурации ---
setup_page()
show_splash()
init_api_key()
init_session()
init_page_state()
setup_sidebar(set_page)

# ===================== СТРАНИЦЫ =======================
# === Загрузка данных ===
if st.session_state['page'] == "Загрузка данных":
    st.title("📥 Загрузка данных")

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
            try:
                update_context("data_summary", summary)
            except Exception:
                pass
        else:
            summary = st.session_state.get(
                "data_summary",
                f"{df.shape[0]} строк, {df.shape[1]} столбцов; признаки: {', '.join(map(str, df.columns))}"
            )

        st.markdown("---")
        # Блок подключения ИИ в экспандере
        with st.expander("🤖 Подключение ИИ", expanded=False):
            st.caption("При желании укажите цель анализа — ИИ адаптирует помощь под неё.")

            user_desc = st.text_area(
                label="Цель анализа",
                placeholder="Например: Хочу проанализировать, как меняются цены на жильё по регионам",
                value=st.session_state.get("analysis_goal", ""),
                height=100,
                label_visibility="collapsed",
                key="analysis_goal_input" 
            )

            if st.button("Подключить ИИ"):
                msg = notify_ai_dataset_and_goal(df, user_desc, get_chatgpt_response)
                st.success(msg)

# === Автообработка данных ===
if st.session_state['page'] == "Автообработка данных":
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
                    if len(clean_log) == 0:
                        st.info("ℹ️ Пропусков не найдено — очистка не потребовалась.")
                    else:
                        st.success("✅ Пропуски успешно обработаны")
                        st.write(pd.DataFrame(clean_log))
                except Exception as e:
                    st.error(f"Ошибка при обработке пропусков: {e}", icon="🚫")

            # Шаг 2: автообработка выбросов
            with st.spinner("Шаг 2/2: обработка выбросов…"):
                time.sleep(2)  # имитация процесса
                try:
                    before_df, outlier_log, df = run_auto_outlier_removal(df)
                    if len(outlier_log) == 0:
                        st.info("ℹ️ Выбросов не обнаружено — обработка не потребовалась.")
                    else:
                        st.success("✅ Выбросы успешно обработаны")
                except Exception as e:
                    st.error(f"Ошибка при обработке выбросов: {e}", icon="🚫")

            # Сохраняем результат
            st.session_state["df"] = df

# === обработка пропусков ===
if st.session_state.get("page") == "Обработка пропусков":
    st.title("⚙️ Обработка пропусков")
    st.caption('Обработка пропущенных значений (NaN)"!')

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
                before, log, new_df = run_auto_cleaning(df, target_col=target)
                st.session_state["df"] = new_df
                st.session_state["data_changed"] = True

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

                    with st.spinner("Автоочистка..."):
                        time.sleep(1)

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

# === Обработка выбросов ===
if st.session_state.get("page") == "Обработка выбросов":
    st.title("🚩 Обработка выбросов")

        # Инициализация флага изменений
    if "data_changed" not in st.session_state:
        st.session_state["data_changed"] = False

    if "df" not in st.session_state:
        st.warning("📥 Загрузите данные на предыдущей странице", icon="⚠️")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        # # Инструкция
        # render_outlier_handling_info()
        # st.markdown("---")

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
            before, log, cleaned_df = run_auto_outlier_removal(df)
            st.session_state["df"] = cleaned_df

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
                fig_cmp = plot_outlier_removal_comparison(df, cleaned_df, numeric_cols)
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

# === Визуализация и EDA ===
elif st.session_state["page"] == "Визуальный анализ":
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
            
# === Сводные таблицы ===
elif st.session_state["page"] == "Сводные таблицы":
    st.title("📟 Сводные таблицы (Pivot)")
    st.caption("ℹ В этом разделе вы можете строить сводные таблицы и визуализировать их.")

    if "df" not in st.session_state:
        st.warning("📥 Сначала загрузите данные.", icon="⚠️")
    else:
        df = st.session_state["df"]
        show_pivot_tab(df)

# === Сравнение групп ===
if st.session_state.get("page") == "Сравнение групп":
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

# === Моделирование и предсказание ===
if st.session_state.get("page") == "Логистическая регрессия":
    st.title("📈 Логистическая регрессия")
    st.caption("ℹ Фокус: понять, как и почему признаки влияют на целевую переменную")

    if "df" not in st.session_state:
        st.warning("📥 Сначала загрузите данные.")
        st.stop()

    df = st.session_state["df"]
    ms = ensure_modeling_state(df)

    options = list(df.columns)
    target_col, _ = sticky_selectbox("modeling_state", "target", "🎯 Целевая переменная (binary target)", options, ui_key="modeling_target_ui")

    if len(pd.Series(df[target_col].dropna().unique())) > 2:
        st.error("Целевая переменная должна быть бинарной")
        st.stop()

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        st.error("Нет признаков для обучения")
        st.stop()

    C_value, penalty, max_iter, threshold, test_size, use_class_weight = show_model_settings()


    if st.button("🚀 Обучить / переобучить модель", use_container_width=True):
        try:
            with st.spinner("⏳ Обучение модели..."):
                time.sleep(5)

                # Подготовка данных
                X, y_encoded, le, num_cols, cat_cols = prepare_features_and_target(df, target_col)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )

                # Обучение
                class_weight = "balanced" if use_class_weight else None
                model, meta = train_logistic_regression(
                    X_train, y_train,
                    C=C_value, penalty=penalty,
                    class_weight=class_weight, max_iter=max_iter,
                    label_encoder=le
                )

                # Оценка
                metrics, roc_data, pr_data = evaluate_model(model, X_test, y_test, meta, threshold)
                importance_df = compute_feature_importance(model, meta)
                short_text = interpret_feature_importance(importance_df, top_n=3)

                # Сохраняем в сессию
                st.session_state["modeling"] = {
                    "model": model, "meta": meta,
                    "threshold": threshold, "metrics": metrics,
                    "roc": roc_data, "pr": pr_data,
                    "importance_df": importance_df, "short_text": short_text,
                    "target_col": target_col, "feature_cols": feature_cols,
                    "params": {
                        "C": C_value, "penalty": penalty,
                        "class_weight": class_weight, "max_iter": max_iter,
                        "test_size": test_size
                    }
                }

                mark_model_trained()

            st.success("✅ Модель обучена и сохранена")

        except Exception as e:
            st.error(f"Не удалось обучить модель: {e}")

    # Если модель уже обучена — показываем результаты
    if "modeling" in st.session_state:
        data = st.session_state["modeling"]

        show_results_and_analysis(data)
        show_single_prediction(data, df)
        show_export_buttons(data)

# === Моделирование CatBoost ===
if st.session_state.get("page") == "CatBoost моделирование":
    st.title("CatBoost моделирование")

    # --- Страховка от отсутствия df ---
    df = st.session_state.get("df")
    if df is None or df.empty:
        st.warning("📥 Сначала загрузите данные.")
        st.stop()

    # --- Инициализация состояния ---
    if "catboost_state" not in st.session_state:
        st.session_state["catboost_state"] = {}

    # --- Выбор целевой переменной ---
    options = list(df.columns)
    target_col = st.selectbox("🎯 Целевая переменная", options)
    if not target_col:
        st.error("❌ Не выбрана целевая переменная")
        st.stop()

    # --- Определение задачи ---
    task = detect_task(df, target_col)
    st.info(f"Задача: {task}")

    # --- Настройки модели ---
    with st.expander("⚙️ Настройки модели", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            iterations = st.slider("Iterations", 100, 3000, 800, step=50)
            depth = st.slider("Depth", 2, 10, 6)
            learning_rate = st.slider("Learning rate", 0.005, 0.2, 0.05)
            l2_leaf_reg = st.slider("L2 leaf reg", 1.0, 10.0, 3.0)
        with col2:
            subsample = st.slider("Subsample", 0.3, 1.0, 0.8)
            colsample_bylevel = st.slider("Colsample by level", 0.3, 1.0, 0.8)
            min_data_in_leaf = st.slider("Min data in leaf", 1, 100, 20)
            test_size = st.slider("Test size", 0.1, 0.5, 0.2)

        threshold = st.slider("Threshold (binary only)", 0.05, 0.95, 0.5) if task == "binary" else 0.5

        available_metrics = valid_eval_metrics_for_task(task)
        eval_metric = st.selectbox("Eval metric", available_metrics, index=0)

        use_recall_monitor = st.checkbox("Мониторить Recall (custom_metric)", value=(task == "binary"))

        use_class_weight = st.checkbox("Баланс классов (binary)", value=(task == "binary"))
        class_weights = None
        if use_class_weight and task == "binary":
            y_tmp = df[target_col]
            if not pd.api.types.is_numeric_dtype(y_tmp):
                y_tmp = pd.factorize(y_tmp)[0]
            if len(np.unique(pd.Series(y_tmp).dropna())) == 2:
                pos_rate = float((y_tmp == 1).mean())
                auto_w = round(1.0 / max(pos_rate, 1e-6), 2)
                st.caption(f"Автовес положительного класса ≈ {auto_w}")
                class_weights = [1.0, auto_w]

    # --- Кнопка обучения ---
    if st.button("🚀 Обучить модель CatBoost", use_container_width=True):
        try:
            with st.spinner("⏳ Обучение модели..."):
                from sklearn.model_selection import train_test_split

                X, y, cat_features = prepare_features_and_target_catboost(df, target_col)
                stratify = y if task in ("binary", "multiclass") else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=stratify
                )

                custom_metric = ["Recall"] if (task == "binary" and use_recall_monitor) else None
                model = train_catboost_universal(
                    X_train, y_train, X_test, y_test, cat_features,
                    task=task,
                    iterations=iterations,
                    depth=depth,
                    lr=learning_rate,
                    l2_leaf_reg=l2_leaf_reg,
                    subsample=subsample,
                    colsample_bylevel=colsample_bylevel,
                    min_data_in_leaf=min_data_in_leaf,
                    class_weights=class_weights,
                    eval_metric=eval_metric,
                    custom_metric=custom_metric,
                )

                metrics, y_pred, y_proba, viz = evaluate_catboost_universal(
                    model, X_test, y_test, task=task, threshold=threshold
                )

                imp_df = compute_catboost_feature_importance(model, X.columns.tolist(), signed=True)
                imp_figs = plot_feature_importance_signed(imp_df, top_n=15)

                cm_fig = None
                if task in ("binary", "multiclass") and y_pred is not None:
                    cm_fig = build_confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

                st.session_state["catboost_state"] = {
                    "model": model,
                    "metrics": metrics,
                    "viz": viz,
                    "importance_df": imp_df,
                    "importance_figs": imp_figs,
                    "confusion_matrix": cm_fig,
                    "target_col": target_col,
                    "feature_cols": X.columns.tolist(),
                    "threshold": threshold,
                    "task": task,
                    # сохраняем индексы категориальных признаков
                    "cat_features_idx": st.session_state.get("cat_features_idx", []),
                    "params": {
                        "iterations": iterations,
                        "depth": depth,
                        "learning_rate": learning_rate,
                        "l2_leaf_reg": l2_leaf_reg,
                        "subsample": subsample,
                        "colsample_bylevel": colsample_bylevel,
                        "min_data_in_leaf": min_data_in_leaf,
                        "test_size": test_size,
                        "class_weights": class_weights,
                        "eval_metric": eval_metric,
                        "use_recall_monitor": use_recall_monitor,
                    }
                }


            st.success("✅ CatBoost модель обучена и сохранена")

        except Exception as e:
            st.error(f"❌ Не удалось обучить модель: {e}")

    # --- Если модель уже обучена ---
    state = st.session_state.get("catboost_state")
    if state:
        tabs = st.tabs(["📊 Результаты модели", "🔮 Прогноз нового объекта"])

        # --- Вкладка 1: результаты ---
        with tabs[0]:
            with st.expander("📊 Метрики (таблица)", expanded=False):
                # Форматируем значения метрик как строки с 3 знаками после запятой
                metrics_df = pd.DataFrame(
                    [(k, f"{v:.3f}") for k, v in state["metrics"].items()],
                    columns=["Метрика", "Значение"]
                )
                st.table(metrics_df)

            if state["task"] == "binary" and "roc_fig" in state["viz"] and "pr_fig" in state["viz"]:
                with st.expander("📈 Визуализация метрик (ROC и PR)", expanded=True):
                    st.plotly_chart(state["viz"]["roc_fig"], use_container_width=True)
                    st.plotly_chart(state["viz"]["pr_fig"], use_container_width=True)

            if state.get("confusion_matrix") is not None:
                with st.expander("🧩 Confusion Matrix", expanded=True):
                    st.plotly_chart(state["confusion_matrix"], use_container_width=True)

            with st.expander("🔥 Важность признаков (топ-15)", expanded=True):
                colp, coln = st.columns(2)
                with colp:
                    fig_pos = state["importance_figs"].get("pos")
                    if fig_pos:
                        st.plotly_chart(fig_pos, use_container_width=True)
                with coln:
                    fig_neg = state["importance_figs"].get("neg")
                    if fig_neg:
                        st.plotly_chart(fig_neg, use_container_width=True)

        # --- Вкладка 2: прогноз нового объекта ---
        with tabs[1]:
            st.subheader("🔮 Прогнозирование нового объекта")

            feature_inputs = {}
            cols = st.columns(2)

            for i, col in enumerate(state["feature_cols"]):
                if pd.api.types.is_numeric_dtype(df[col]):
                    with cols[i % 2]:
                        feature_inputs[col] = st.number_input(f"{col}", value=float(df[col].median()))
                else:
                    with cols[i % 2]:
                        options = df[col].dropna().unique().tolist()
                        feature_inputs[col] = st.selectbox(f"{col}", options)

            import time
            import pandas as pd
            import plotly.express as px

            output_area = st.empty()

            if st.button("📌 Сделать прогноз", use_container_width=True):
                with st.spinner("⏳ Модель делает прогноз..."):
                    time.sleep(1.5)

                    try:
                        result = predict_new_object(
                            state["model"], feature_inputs,
                            task=state["task"], threshold=state["threshold"]
                        )
                        st.session_state["last_prediction"] = result

                        with output_area.container():
                            st.success("✅ Прогноз готов")

                            if state["task"] == "binary":
                                st.write(f"**Предсказанный класс:** {result['prediction']}")
                                st.write(f"**Вероятность положительного класса:** {result['probability']:.3f}")

                                # Визуализация вероятности
                                fig = px.bar(
                                    x=["Negative", "Positive"],
                                    y=[1 - result["probability"], result["probability"]],
                                    labels={"x": "Класс", "y": "Вероятность"},
                                    title="Вероятности классов",
                                    color=["Negative", "Positive"],
                                    color_discrete_map={"Negative": "steelblue", "Positive": "crimson"}
                                )
                                st.plotly_chart(fig, use_container_width=True, key="binary_probs")

                            elif state["task"] == "multiclass":
                                st.write(f"**Предсказанный класс:** {result['prediction']}")
                                st.write("**Вероятности по классам:**")

                                # Красивый бар-чарт для вероятностей
                                fig = px.bar(
                                    x=list(range(len(result["probabilities"]))),
                                    y=result["probabilities"],
                                    labels={"x": "Класс", "y": "Вероятность"},
                                    title="Вероятности по классам",
                                    color=result["probabilities"],
                                    color_continuous_scale="Viridis"
                                )
                                st.plotly_chart(fig, use_container_width=True, key="multiclass_probs")

                            else:  # regression
                                st.write(f"**Предсказанное значение:** {result['prediction']:.3f}")

                            # --- Важность признаков (из обучения) ---
                            st.markdown("### Признаки которые влияли на прогноз")
                            fig_pos = state["importance_figs"].get("pos")
                            fig_neg = state["importance_figs"].get("neg")

                            if fig_pos:
                                st.plotly_chart(fig_pos, use_container_width=True, key="feat_imp_pos")
                            if fig_neg:
                                st.plotly_chart(fig_neg, use_container_width=True, key="feat_imp_neg")

                    except Exception as e:
                        st.error(f"❌ Ошибка при прогнозировании: {e}")

# === Разъяснение результатов (с ИИ) ===
if st.session_state.get("page") == "Разъяснение результатов (с ИИ)":
    st.title("💬 Поговорим о ваших данных?")
    st.markdown("---")

    if st.button("🗑 Очистить чат"):
        reset_chat_history()
        st.success("Чат очищен.")
        st.stop()

    st.session_state.setdefault("chat_history", [])

    # Ввод нового сообщения
    question = st.chat_input("Напишите свой вопрос…")

    if question:
        # Добавляем вопрос в историю
        st.session_state.chat_history.append({"text": question, "sender": "user"})

        # Сначала рендерим всю историю (включая новый вопрос)
        for msg in st.session_state.chat_history:
            render_message(msg["text"], msg["sender"])

        # Временный индикатор "ИИ печатает..."
        placeholder = st.empty()
        placeholder.markdown(
            """
            <style>
            @keyframes blink {
                0%   { opacity: 0.2; }
                20%  { opacity: 1; }
                100% { opacity: 0.2; }
            }
            .dot {
                display: inline-block;
                margin-left: 2px;
                animation: blink 1.4s infinite both;
            }
            .dot:nth-child(2) { animation-delay: 0.2s; }
            .dot:nth-child(3) { animation-delay: 0.4s; }
            </style>

            <div style='
                background: var(--background-color);
                color: var(--text-color);
                padding: 10px 14px;
                border-radius: 12px;
                text-align: left;
                margin: 6px 0;
                font-style: italic;
                opacity: 0.85;
                box-shadow: 0 1px 3px rgba(0,0,0,0.15);
            '>
                🤖 ИИ печатает<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


        # Получаем ответ ИИ (это занимает время)
        answer = continue_chat(question)

        # Заменяем индикатор на настоящий ответ
        placeholder.empty()
        st.session_state.chat_history.append({"text": answer, "sender": "ai"})
        render_message(answer, "ai")

    else:
        # Если нового вопроса нет — просто рендерим историю
        for msg in st.session_state.chat_history:
            render_message(msg["text"], msg["sender"])

# === Футер внизу страницы (автор) ===
# Постоянная надпись внизу лево, вне зависимости от содержимого
st.markdown("""
    <style>
        .bottom-right {
            position: fixed;
            right: 15px;
            bottom: 10px;
            font-size: 0.75em;
            color: #333333;
            z-index: 9999;
        }
    </style>
    <div class="bottom-right">© Created by Rahimov M.A. TTU 2025</div>
""", unsafe_allow_html=True)