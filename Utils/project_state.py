import io
import pickle
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from Utils.database_utils import (
    download_binary_file,
    download_dataset_file,
    get_analysis_results,
    get_datasets,
    save_analysis_result,
    upload_binary_file,
)
from Utils.upload_utils import load_data


SNAPSHOT_RESULT_TYPE = "project_snapshot"
SKIP_VALUE = object()
RESTORE_SKIP_VALUE = object()

EXCLUDED_STATE_KEYS = {
    "df",
    "current_project_id",
    "current_project_name",
    "pending_dataset_file_bytes",
    "pending_dataset_filename",
    "OPENAI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
}

EXCLUDED_KEY_PREFIXES = (
    "confirm_delete_",
    "delete_project_",
    "open_project_",
    "create_project_form",
)

CRITICAL_STATE_KEYS = {
    "_data_sig",
    "academic_performance_schema",
    "auto_clean_log",
    "auto_outlier_log",
    "auto_proc_done",
    "chat_history",
    "conversion_log",
    "data_changed",
    "data_summary",
    "df_before_outlier",
    "eda_built",
    "eda_chart",
    "eda_chart_index",
    "eda_fig",
    "eda_suggestion",
    "eda_x",
    "eda_x_index",
    "eda_y",
    "eda_y_index",
    "eda_z",
    "eda_z_index",
    "feature_cols",
    "group_comparison_results",
    "last_prediction",
    "modeling",
    "modeling_state",
    "missing_auto_before",
    "missing_auto_done",
    "missing_auto_log",
    "original_filename",
    "outlier_auto_done",
    "outlier_auto_log",
    "pivot_agg",
    "pivot_agg_index",
    "pivot_chart_type",
    "pivot_config",
    "pivot_index_cols",
    "pivot_saved",
    "pivot_table",
    "pivot_value",
    "pivot_value_index",
    "selected_columns",
    "stats_test_choice",
    "stats_test_results",
    "visualization_state",
}


class NamedBytesIO(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def _utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _is_project_state_key(key):
    if key in EXCLUDED_STATE_KEYS:
        return False
    if str(key).startswith(EXCLUDED_KEY_PREFIXES):
        return False
    if key in CRITICAL_STATE_KEYS:
        return True
    return True


def _pickleable(value):
    try:
        pickle.dumps(value)
        return True
    except Exception:
        return False


def _is_plotly_figure(value):
    return value.__class__.__module__.startswith("plotly.") and hasattr(value, "to_json")


def _is_trained_model_like(value):
    module_name = value.__class__.__module__
    return module_name.startswith(("sklearn.", "catboost.", "xgboost.", "lightgbm."))


def _sanitize_for_snapshot(value, key=None):
    if _is_plotly_figure(value):
        try:
            return {"__project_plotly_json__": value.to_json()}
        except Exception:
            return SKIP_VALUE

    if _is_trained_model_like(value):
        return SKIP_VALUE

    if isinstance(value, dict):
        sanitized = {}
        for item_key, item_value in value.items():
            if item_key in {"model", "meta"}:
                continue
            nested = _sanitize_for_snapshot(item_value, key=item_key)
            if nested is not SKIP_VALUE:
                sanitized[item_key] = nested
        return sanitized

    if isinstance(value, list):
        sanitized = []
        for item in value:
            nested = _sanitize_for_snapshot(item)
            if nested is not SKIP_VALUE:
                sanitized.append(nested)
        return sanitized

    if isinstance(value, tuple):
        sanitized = []
        for item in value:
            nested = _sanitize_for_snapshot(item)
            if nested is not SKIP_VALUE:
                sanitized.append(nested)
        return tuple(sanitized)

    if isinstance(value, set):
        sanitized = []
        for item in value:
            nested = _sanitize_for_snapshot(item)
            if nested is not SKIP_VALUE:
                sanitized.append(nested)
        return sanitized

    if _pickleable(value):
        return value

    return SKIP_VALUE


def _restore_snapshot_value(value):
    if isinstance(value, dict):
        if "__project_plotly_json__" in value:
            try:
                import plotly.io as pio

                return pio.from_json(value["__project_plotly_json__"])
            except Exception:
                return RESTORE_SKIP_VALUE

        restored = {}
        for item_key, item_value in value.items():
            restored_value = _restore_snapshot_value(item_value)
            if restored_value is not RESTORE_SKIP_VALUE:
                restored[item_key] = restored_value
        return restored

    if isinstance(value, list):
        return [_restore_snapshot_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_restore_snapshot_value(item) for item in value)

    return value


def _resolve_source_dataset(project_id):
    current_dataset_id = st.session_state.get("current_dataset_id")
    if current_dataset_id:
        return current_dataset_id

    datasets = [
        item for item in get_datasets(project_id)
        if not str(item.get("original_filename") or "").startswith("project_snapshot_")
    ]
    if datasets:
        return datasets[0].get("id")

    return None


def collect_project_state():
    state = {}
    skipped = []

    for key in list(st.session_state.keys()):
        if not _is_project_state_key(key):
            continue

        value = st.session_state.get(key)
        sanitized_value = _sanitize_for_snapshot(value, key=key)
        if sanitized_value is SKIP_VALUE or not _pickleable(sanitized_value):
            skipped.append(key)
            continue
        state[key] = sanitized_value

    return state, skipped


def save_current_project_state(project_id, project_name=None):
    if not project_id:
        return None, "Сначала откройте или создайте проект."
    if "df" not in st.session_state or st.session_state["df"] is None:
        return None, "Сначала загрузите датасет."

    df = st.session_state["df"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, "Текущий датасет пустой, сохранять нечего."

    source_dataset_id = _resolve_source_dataset(project_id)
    if not source_dataset_id:
        return None, "Сначала загрузите или сохраните датасет в проект, затем сохраняйте снимок проекта."

    stamp = _utc_stamp()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    df_path = f"project_states/{project_id}/{stamp}_data.csv"
    uploaded_df_path = upload_binary_file(
        csv_bytes,
        df_path,
        content_type="text/csv",
    )
    if not uploaded_df_path:
        return None, "Не удалось сохранить текущую версию датасета в снимок проекта."

    state, skipped = collect_project_state()
    payload = {
        "version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "project_name": project_name,
        "source_dataset_id": source_dataset_id,
        "df_path": uploaded_df_path,
        "state": state,
    }

    snapshot_bytes = pickle.dumps(payload)
    snapshot_path = f"project_states/{project_id}/{stamp}_state.pkl"
    uploaded_snapshot_path = upload_binary_file(
        snapshot_bytes,
        snapshot_path,
        content_type="application/octet-stream",
    )
    if not uploaded_snapshot_path:
        return None, "Датасет сохранен, но снимок состояния приложения сохранить не удалось."

    metadata = {
        "saved_at": payload["saved_at"],
        "snapshot_path": uploaded_snapshot_path,
        "source_dataset_id": source_dataset_id,
        "df_path": uploaded_df_path,
        "rows_count": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "state_keys": sorted(state.keys()),
        "skipped_state_keys": skipped,
    }
    saved_result = save_analysis_result(source_dataset_id, SNAPSHOT_RESULT_TYPE, metadata)
    if not saved_result:
        return None, "Снимок загружен, но метаданные снимка не сохранились."

    return metadata, None


def get_latest_project_snapshot(project_id):
    if not project_id:
        return None

    snapshots = []
    for dataset in get_datasets(project_id):
        dataset_id = dataset.get("id")
        for item in get_analysis_results(dataset_id, SNAPSHOT_RESULT_TYPE):
            result_json = item.get("result_json") or {}
            snapshots.append(
                {
                    "analysis_result": item,
                    "dataset": dataset,
                    "saved_at": result_json.get("saved_at") or item.get("created_at"),
                    "snapshot_path": result_json.get("snapshot_path"),
                    "df_path": result_json.get("df_path"),
                    "source_dataset_id": result_json.get("source_dataset_id") or dataset_id,
                }
            )

    snapshots = [item for item in snapshots if item.get("snapshot_path")]
    if not snapshots:
        return None

    return sorted(snapshots, key=lambda item: item.get("saved_at") or "", reverse=True)[0]


def restore_project_snapshot(project_id, project_name=None):
    latest = get_latest_project_snapshot(project_id)
    if not latest:
        return None, "Для этого проекта пока нет сохраненного снимка."

    dataset = latest["dataset"]
    df_path = latest.get("df_path")
    file_bytes = download_binary_file(df_path) if df_path else download_dataset_file(dataset.get("file_path"))
    if not file_bytes:
        return None, "Не удалось загрузить сохраненный датасет."

    snapshot_bytes = download_binary_file(latest["snapshot_path"])
    if not snapshot_bytes:
        return None, "Не удалось загрузить состояние проекта."

    try:
        state_payload = pickle.loads(snapshot_bytes)
        file_obj = NamedBytesIO(file_bytes, dataset.get("original_filename") or "dataset.csv")
        df = load_data(file_obj)
    except Exception as exc:
        return None, f"Не удалось восстановить проект: {exc}"

    restored_state = state_payload.get("state", {})
    for key, value in restored_state.items():
        st.session_state[key] = _restore_snapshot_value(value)

    st.session_state["df"] = df
    st.session_state["current_project_id"] = project_id
    st.session_state["current_project_name"] = project_name or state_payload.get("project_name")
    st.session_state["current_dataset_id"] = (
        state_payload.get("source_dataset_id")
        or latest.get("source_dataset_id")
        or dataset.get("id")
    )
    st.session_state["original_filename"] = (
        restored_state.get("original_filename")
        or dataset.get("original_filename")
    )
    st.session_state.pop("pending_dataset_file_bytes", None)
    st.session_state.pop("pending_dataset_filename", None)

    return {
        "dataset": dataset,
        "snapshot": latest,
        "state_keys": sorted(restored_state.keys()),
    }, None
