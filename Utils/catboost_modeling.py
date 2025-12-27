# Utils/catboost_modeling.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


# -----------------------------
# Task detection (unchanged)
# -----------------------------
def detect_task(df: pd.DataFrame, target_col: str) -> str:
    try:
        y = df[target_col]
        if pd.api.types.is_numeric_dtype(y):
            unique_vals = y.dropna().unique()
            if pd.api.types.is_integer_dtype(y) and len(unique_vals) <= 20:
                return "multiclass" if len(unique_vals) > 2 else "binary"
            return "regression"
        else:
            unique_vals = y.dropna().unique()
            return "multiclass" if len(unique_vals) > 2 else "binary"
    except Exception:
        # Fallback: if any issue, assume regression to avoid classification constraints
        return "regression"


# -----------------------------
# Safety hints
# -----------------------------
def make_hint(message: str) -> str:
    """
    Формирует короткую подсказку для пользователя в UI.
    """
    return f"ℹ Подсказка: {message}"


# -----------------------------
# Target/data preparation with validation
# -----------------------------
def prepare_features_and_target_catboost(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Универсальная подготовка:
    - Автокодирование категориального таргета
    - Определение категориальных признаков
    - Проверка пустых данных и константного таргета
    """
    if target_col not in df.columns:
        raise ValueError("Целевая переменная не найдена в данных.")

    if df.empty:
        raise ValueError("Датасет пустой.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Проверки целевой
    if y.dropna().nunique() < 2:
        raise ValueError("Целевая переменная должна иметь минимум 2 уникальных значения.")

    if not pd.api.types.is_numeric_dtype(y):
        classes = y.dropna().unique()
        mapping = {cls: i for i, cls in enumerate(sorted(classes))}
        y = y.map(mapping)

    # Категориальные признаки
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Если есть полностью пустые колонки/строки, подскажем
    if X.isnull().all(axis=0).any():
        raise ValueError(make_hint("Есть признаки с одними пропусками. Удалите или заполните пропуски."))

    if X.isnull().all(axis=1).any():
        raise ValueError(make_hint("Есть строки, где все признаки пропущены. Очистите данные."))

    return X, y, cat_features


def compute_catboost_feature_importance(model, feature_names: List[str], signed: bool = True) -> pd.DataFrame:
    try:
        if signed:
            importances = model.get_feature_importance(type="PredictionValuesChange")
        else:
            importances = model.get_feature_importance()
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        return df_imp
    except Exception as e:
        raise RuntimeError(f"Ошибка важности признаков: {e}")



# -----------------------------
# Valid eval metrics by task
# -----------------------------
def valid_eval_metrics_for_task(task: str) -> List[str]:
    if task == "binary":
        return ["Logloss", "Accuracy", "AUC", "F1"]
    elif task == "multiclass":
        return ["MultiClass", "Accuracy", "F1"]
    else:
        return ["RMSE", "MAE", "R2"]


# -----------------------------
# Build CatBoost with extended medical-focused hyperparams
# -----------------------------
def build_catboost(
    task: str,
    iterations: int = 800,
    depth: int = 6,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 3.0,
    subsample: Optional[float] = None,  # по умолчанию None
    colsample_bylevel: float = 0.8,
    min_data_in_leaf: int = 20,
    class_weights: Optional[Union[List[float], Dict[int, float]]] = None,
    eval_metric: Optional[str] = None,
    custom_metric: Optional[List[str]] = None,
    random_seed: int = 42
):
    try:
        if task in ("binary", "multiclass"):
            params = {
                "iterations": iterations,
                "depth": depth,
                "learning_rate": learning_rate,
                "l2_leaf_reg": l2_leaf_reg,
                "colsample_bylevel": colsample_bylevel,
                "min_data_in_leaf": min_data_in_leaf,
                "loss_function": "Logloss" if task == "binary" else "MultiClass",
                "random_seed": random_seed,
                "verbose": False
            }
            # subsample работает только с Bernoulli
            if subsample is not None:
                params["bootstrap_type"] = "Bernoulli"
                params["subsample"] = subsample

            if eval_metric is None:
                params["eval_metric"] = "AUC" if task == "binary" else "MultiClass"
            else:
                if eval_metric not in valid_eval_metrics_for_task(task):
                    raise ValueError(make_hint("Выбрана неподходящая eval_metric для задачи."))
                params["eval_metric"] = eval_metric

            if custom_metric and task == "binary":
                params["custom_metric"] = custom_metric

            if class_weights is not None:
                params["class_weights"] = class_weights

            return CatBoostClassifier(**params)

        else:  # regression
            params = {
                "iterations": iterations,
                "depth": depth,
                "learning_rate": learning_rate,
                "l2_leaf_reg": l2_leaf_reg,
                "colsample_bylevel": colsample_bylevel,
                "min_data_in_leaf": min_data_in_leaf,
                "loss_function": "RMSE",
                "random_seed": random_seed,
                "verbose": False
            }
            if subsample is not None:
                params["bootstrap_type"] = "Bernoulli"
                params["subsample"] = subsample

            if eval_metric and eval_metric in valid_eval_metrics_for_task("regression"):
                params["eval_metric"] = eval_metric

            return CatBoostRegressor(**params)

    except Exception as e:
        raise ValueError(f"Ошибка конфигурации модели: {e}")


# -----------------------------
# Train CatBoost universally (with try/except)
# -----------------------------
def train_catboost_universal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_features: List[str],
    task: str,
    iterations: int = 800,
    depth: int = 6,
    lr: float = 0.05,
    l2_leaf_reg: float = 3.0,
    subsample: Optional[float] = None,
    colsample_bylevel: float = 0.8,
    min_data_in_leaf: int = 20,
    class_weights: Optional[Union[List[float], Dict[int, float]]] = None,
    eval_metric: Optional[str] = None,
    custom_metric: Optional[List[str]] = None,
    random_seed: int = 42
):
    try:
        # Приводим категориальные признаки к строкам
        for col in cat_features:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)

        # Получаем индексы категориальных признаков
        cat_features_idx = [X_train.columns.get_loc(col) for col in cat_features]

        # Строим модель
        model = build_catboost(
            task=task, iterations=iterations, depth=depth,
            learning_rate=lr, l2_leaf_reg=l2_leaf_reg,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel, min_data_in_leaf=min_data_in_leaf,
            class_weights=class_weights, eval_metric=eval_metric, custom_metric=custom_metric,
            random_seed=random_seed
        )

        # Обучаем модель
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features_idx
        )

        # Сохраняем индексы категориальных признаков в session_state
        import streamlit as st
        st.session_state["cat_features_idx"] = cat_features_idx

        return model
    except Exception as e:
        hint = make_hint("Проверьте корректность метрик, весов классов и наличие пропусков в данных.")
        raise RuntimeError(f"Не удалось обучить модель: {e}\n{hint}")



# -----------------------------
# Evaluation (with safe guards)
# -----------------------------
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    root_mean_squared_error
)

def evaluate_catboost_universal(
    model, X_test: pd.DataFrame, y_test: pd.Series, task: str, threshold: float = 0.5
) -> Tuple[Dict[str, float], np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    metrics: Dict[str, float] = {}
    y_pred = None
    y_proba = None
    viz: Dict[str, Any] = {}

    try:
        if task == "binary":
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

            if y_proba is not None:
                y_pred = (y_proba >= threshold).astype(int)
            else:
                y_pred = model.predict(X_test)

            metrics = {
                "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
                "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
                "F1": round(f1_score(y_test, y_pred, zero_division=0), 3)
            }
            if y_proba is not None:
                metrics["ROC-AUC"] = round(roc_auc_score(y_test, y_proba), 3)
                metrics["PR-AUC"] = round(average_precision_score(y_test, y_proba), 3)

            if y_proba is not None:
                viz["roc_fig"] = build_roc_curve(y_test, y_proba)
                viz["pr_fig"] = build_pr_curve(y_test, y_proba)

        elif task == "multiclass":
            y_pred = model.predict(X_test)
            metrics = {
                "Accuracy": round(accuracy_score(y_test, y_pred), 3),
                "Macro-Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 3),
                "Macro-Recall": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 3),
                "Macro-F1": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 3),
            }

        else:  # regression
            y_pred = model.predict(X_test)
            metrics = {
                "RMSE": round(root_mean_squared_error(y_test, y_pred), 3),
                "MSE": round(mean_squared_error(y_test, y_pred), 3),
                "MAE": round(mean_absolute_error(y_test, y_pred), 3),
                "R2": round(r2_score(y_test, y_pred), 3)
            }

    except Exception as e:
        hint = make_hint("Проверьте формат целевой переменной и совместимость выбранной метрики.")
        raise RuntimeError(f"Не удалось оценить модель: {e}\n{hint}")

    return metrics, y_pred, y_proba, viz




# -----------------------------
# Plotly builders (unchanged)
# -----------------------------
def build_metrics_table(metrics: Dict[str, float]) -> go.Figure:
    items = list(metrics.items())
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Metric", "Value"], fill_color="#f0f0f0", align="left"),
        cells=dict(values=[[k for k, _ in items], [f"{v:.3f}" for _, v in items]], align="left")
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
    return fig


def build_roc_curve(y_true: Union[np.ndarray, pd.Series], y_score: np.ndarray) -> go.Figure:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", height=380)
    return fig


def build_pr_curve(y_true: Union[np.ndarray, pd.Series], y_score: np.ndarray) -> go.Figure:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AP={ap:.3f})"))
    fig.update_layout(title="Precision–Recall curve", xaxis_title="Recall", yaxis_title="Precision", height=380)
    return fig


# -----------------------------
# Confusion Matrix
# -----------------------------
def build_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = np.unique(y_true)

    z = cm.astype(int)
    x = [str(l) for l in labels]
    y = [str(l) for l in labels]

    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale="Blues", showscale=True)
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True", height=420)
    return fig


# -----------------------------
# Feature importance (signed) and plotting (unchanged)
# -----------------------------
def compute_catboost_feature_importance(
    model, feature_names: List[str], signed: bool = True
) -> pd.DataFrame:
    try:
        if signed:
            importances = model.get_feature_importance(type="PredictionValuesChange")
        else:
            importances = model.get_feature_importance()
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        return df_imp
    except Exception as e:
        hint = make_hint("Не удалось вычислить важность признаков. Проверьте обученную модель.")
        raise RuntimeError(f"Ошибка важности признаков: {e}\n{hint}")


def plot_feature_importance_signed(imp_df: pd.DataFrame, top_n: int = 10) -> Dict[str, go.Figure]:
    pos_df = imp_df[imp_df["importance"] > 0].nlargest(top_n, "importance")
    neg_df = imp_df[imp_df["importance"] < 0].nsmallest(top_n, "importance")

    figs = {}
    fig_pos = px.bar(
        pos_df.iloc[::-1], x="importance", y="feature", orientation="h", title="Top feature effects"
    )
    fig_pos.update_layout(height=420, xaxis_title="Importance (+)", yaxis_title="")
    figs["pos"] = fig_pos

    return figs


from catboost import Pool

def predict_new_object(model, feature_values: dict, task: str, threshold: float = 0.5):
    import pandas as pd
    import streamlit as st

    X_new = pd.DataFrame([feature_values])

    cat_idx = st.session_state.get("cat_features_idx", [])
    feature_cols = st.session_state.get("feature_cols", X_new.columns.tolist())

    # Приводим типы
    for i, col in enumerate(feature_cols):
        if i in cat_idx:
            X_new[col] = X_new[col].astype(str)
        else:
            X_new[col] = pd.to_numeric(X_new[col], errors="coerce")

    # Создаём Pool
    X_new_pool = Pool(data=X_new, cat_features=cat_idx)

    if task == "binary":
        proba = model.predict_proba(X_new_pool)[:, 1][0]
        pred = int(proba >= threshold)
        return {"prediction": pred, "probability": proba}

    elif task == "multiclass":
        proba = model.predict_proba(X_new_pool)[0]
        pred = int(model.predict(X_new_pool)[0])
        return {"prediction": pred, "probabilities": proba.tolist()}

    else:  # regression
        pred = float(model.predict(X_new_pool)[0])
        return {"prediction": pred}

def explain_prediction(model, X_new: pd.DataFrame, feature_names: list, top_n: int = 5):
    import pandas as pd
    import numpy as np
    from catboost import Pool
    import streamlit as st

    cat_idx = st.session_state.get("cat_features_idx", [])
    feature_cols = st.session_state.get("feature_cols", X_new.columns.tolist())

    # Приводим типы
    for i, col in enumerate(feature_cols):
        if i in cat_idx:
            X_new[col] = X_new[col].astype(str)
        else:
            X_new[col] = pd.to_numeric(X_new[col], errors="coerce")

    # Создаём Pool
    X_new_pool = Pool(data=X_new, cat_features=cat_idx)

    importances = model.get_feature_importance(type="PredictionValuesChange", data=X_new_pool)
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", key=np.abs, ascending=False).head(top_n)

    return imp_df
