import io
import pickle
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from Utils.database_utils import (
    create_dataset_record,
    download_binary_file,
    download_dataset_file,
    get_analysis_results,
    get_datasets,
    save_analysis_result,
    upload_binary_file,
    upload_dataset_file,
)
from Utils.upload_utils import load_data


SNAPSHOT_RESULT_TYPE = "project_snapshot"

EXCLUDED_STATE_KEYS = {
    "df",
    "current_project_id",
    "current_project_name",
    "current_dataset_id",
    "pending_dataset_file_bytes",
    "pending_dataset_filename",
    "OPENAI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
}

SAVED_KEY_PREFIXES = (
    "auto_",
    "missing_",
    "outlier_",
    "out_",
    "iqr_",
    "z_",
    "percentile_",
    "eda_",
    "slider_",
    "pivot_",
    "stats_",
    "ttest_",
    "anova_",
    "chi_",
    "modeling",
    "catboost",
)

SAVED_EXACT_KEYS = {
    "chat_history",
    "conversion_log",
    "data_changed",
    "data_summary",
    "df_before_outlier",
    "feature_cols",
    "last_prediction",
    "original_filename",
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
    if key in SAVED_EXACT_KEYS:
        return True
    return any(str(key).startswith(prefix) for prefix in SAVED_KEY_PREFIXES)


def _pickleable(value):
    try:
        pickle.dumps(value)
        return True
    except Exception:
        return False


def collect_project_state():
    state = {}
    skipped = []

    for key in list(st.session_state.keys()):
        if not _is_project_state_key(key):
            continue

        value = st.session_state.get(key)
        if _pickleable(value):
            state[key] = value
        else:
            skipped.append(key)

    return state, skipped


def save_current_project_state(project_id, project_name=None):
    if not project_id:
        return None, "Сначала откройте или создайте проект."
    if "df" not in st.session_state or st.session_state["df"] is None:
        return None, "Сначала загрузите датасет."

    df = st.session_state["df"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, "Текущий датасет пустой, сохранять нечего."

    stamp = _utc_stamp()
    source_name = st.session_state.get("original_filename") or "dataset.csv"
    snapshot_filename = f"project_snapshot_{stamp}_{source_name.rsplit('/', 1)[-1]}"
    if not snapshot_filename.lower().endswith(".csv"):
        snapshot_filename = f"{snapshot_filename}.csv"

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    csv_file = NamedBytesIO(csv_bytes, snapshot_filename)
    dataset_path = upload_dataset_file(csv_file, project_id)
    if not dataset_path:
        return None, "Не удалось сохранить текущий датасет в Supabase Storage."

    dataset = create_dataset_record(
        project_id=project_id,
        original_filename=snapshot_filename,
        file_path=dataset_path,
        rows_count=int(df.shape[0]),
        columns_count=int(df.shape[1]),
    )
    if not dataset:
        return None, "Датасет загружен, но запись о нем не сохранилась в базе."

    state, skipped = collect_project_state()
    payload = {
        "version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "project_name": project_name,
        "dataset_id": dataset["id"],
        "dataset_path": dataset_path,
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
        "dataset_id": dataset["id"],
        "dataset_path": dataset_path,
        "rows_count": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "state_keys": sorted(state.keys()),
        "skipped_state_keys": skipped,
    }
    saved_result = save_analysis_result(dataset["id"], SNAPSHOT_RESULT_TYPE, metadata)
    if not saved_result:
        return None, "Снимок загружен, но метаданные снимка не сохранились."

    st.session_state["current_dataset_id"] = dataset["id"]
    st.session_state["original_filename"] = snapshot_filename

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
    file_bytes = download_dataset_file(dataset.get("file_path"))
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
        st.session_state[key] = value

    st.session_state["df"] = df
    st.session_state["current_project_id"] = project_id
    st.session_state["current_project_name"] = project_name or state_payload.get("project_name")
    st.session_state["current_dataset_id"] = dataset.get("id")
    st.session_state["original_filename"] = dataset.get("original_filename")
    st.session_state.pop("pending_dataset_file_bytes", None)
    st.session_state.pop("pending_dataset_filename", None)

    return {
        "dataset": dataset,
        "snapshot": latest,
        "state_keys": sorted(restored_state.keys()),
    }, None
