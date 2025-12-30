import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
from Utils.AI_helper import connect_ai_model_results
import time as time_module



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
                except Exception as e:
                    st.error(f"❌ Ошибка при прогнозировании: {e}")

        # --- Отображение результата (если есть в сессии) ---
        last_pred = st.session_state.get("last_prediction")
        if last_pred is not None:
             with output_area.container():
                st.success("✅ Прогноз готов")
                
                result = last_pred

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

    # === Кнопка объяснения результатов в ИИ (перед экспортом) ===
    if st.button("🤖 Объяснить результаты в ИИ", key="catboost_ai_explain_bottom", use_container_width=True):
        with st.spinner("⏳ Отправляем результаты..."):
            time_module.sleep(1.0)
            try:
                top_features = state["importance_df"]["feature"].head(5).tolist() if "importance_df" in state else None
            except:
                top_features = None
            connect_ai_model_results(
                metrics=state["metrics"],
                model_type=f"CatBoost ({state['task']})",
                target_col=state["target_col"],
                top_features=top_features
            )
        st.success("✅ Результаты отправлены в ИИ. Перейдите в 'Чат с ИИ' и спросите о метриках!")

    # === Экспорт модели (в самом конце) ===
    st.markdown("---")
    st.subheader("📥 Экспорт модели")
    
    import pickle
    import io
    
    model_bytes = io.BytesIO()
    pickle.dump(state["model"], model_bytes)
    model_bytes.seek(0)
    
    st.download_button(
        label="💾 Скачать модель (.pkl)",
        data=model_bytes,
        file_name="catboost_model.pkl",
        mime="application/octet-stream",
        use_container_width=True
    )
