import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from Utils.modeling_utils import ensure_modeling_state, sticky_selectbox, show_model_settings, \
                                    prepare_features_and_target, train_logistic_regression, evaluate_model, \
                                    compute_feature_importance, interpret_feature_importance, mark_model_trained, \
                                    show_results_and_analysis, show_single_prediction, show_export_buttons



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
            time.sleep(1.5)

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
