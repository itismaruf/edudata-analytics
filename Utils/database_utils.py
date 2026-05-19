import os
import re
from datetime import datetime, timezone

from Utils.supabase_client import get_supabase_client


DATASETS_BUCKET = "datasets"


def _execute_data(query, default=None):
    try:
        response = query.execute()
        return getattr(response, "data", default)
    except Exception:
        return default


def _single(data):
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def _safe_filename(filename: str) -> str:
    base = os.path.basename(filename or "dataset")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or "dataset"


def create_project(name, description=None, user_id=None):
    client = get_supabase_client()
    if client is None or not name:
        return None

    payload = {
        "name": name,
        "description": description,
        "user_id": user_id,
    }
    data = _execute_data(client.table("projects").insert(payload), default=None)
    return _single(data)


def get_projects(user_id=None):
    client = get_supabase_client()
    if client is None:
        return []

    query = client.table("projects").select("*").order("created_at", desc=True)
    if user_id is not None:
        query = query.eq("user_id", user_id)
    return _execute_data(query, default=[]) or []


def get_project(project_id):
    client = get_supabase_client()
    if client is None or not project_id:
        return None

    data = _execute_data(
        client.table("projects").select("*").eq("id", project_id).limit(1),
        default=None,
    )
    return _single(data)


def update_project(project_id, data):
    client = get_supabase_client()
    if client is None or not project_id or not isinstance(data, dict):
        return None

    payload = dict(data)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    result = _execute_data(
        client.table("projects").update(payload).eq("id", project_id),
        default=None,
    )
    return _single(result)


def delete_project(project_id):
    client = get_supabase_client()
    if client is None or not project_id:
        return None

    data = _execute_data(
        client.table("projects").delete().eq("id", project_id),
        default=None,
    )
    return _single(data)


def upload_dataset_file(file, project_id):
    client = get_supabase_client()
    if client is None or file is None or not project_id:
        return None

    filename = _safe_filename(getattr(file, "name", "dataset"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    file_path = f"datasets/{project_id}/{timestamp}_{filename}"

    try:
        if hasattr(file, "getvalue"):
            content = file.getvalue()
        else:
            current_pos = file.tell() if hasattr(file, "tell") else None
            content = file.read()
            if current_pos is not None and hasattr(file, "seek"):
                file.seek(current_pos)

        client.storage.from_(DATASETS_BUCKET).upload(
            path=file_path,
            file=content,
            file_options={"upsert": "false"},
        )
        return file_path
    except Exception:
        return None


def create_dataset_record(project_id, original_filename, file_path, rows_count, columns_count):
    client = get_supabase_client()
    if client is None or not project_id or not original_filename or not file_path:
        return None

    payload = {
        "project_id": project_id,
        "original_filename": original_filename,
        "file_path": file_path,
        "rows_count": rows_count,
        "columns_count": columns_count,
    }
    data = _execute_data(client.table("datasets").insert(payload), default=None)
    return _single(data)


def get_datasets(project_id):
    client = get_supabase_client()
    if client is None or not project_id:
        return []

    return _execute_data(
        client.table("datasets").select("*").eq("project_id", project_id).order("created_at", desc=True),
        default=[],
    ) or []


def get_dataset(dataset_id):
    client = get_supabase_client()
    if client is None or not dataset_id:
        return None

    data = _execute_data(
        client.table("datasets").select("*").eq("id", dataset_id).limit(1),
        default=None,
    )
    return _single(data)


def download_dataset_file(file_path):
    client = get_supabase_client()
    if client is None or not file_path:
        return None

    try:
        return client.storage.from_(DATASETS_BUCKET).download(file_path)
    except Exception:
        return None


def save_educational_schema(dataset_id, schema_dict):
    client = get_supabase_client()
    if client is None or not dataset_id or not isinstance(schema_dict, dict):
        return None

    allowed_fields = {
        "student_col",
        "student_name_col",
        "group_col",
        "course_col",
        "score_col",
        "attendance_col",
        "final_score_col",
        "status_col",
    }
    payload = {field: schema_dict.get(field) for field in allowed_fields}
    payload["dataset_id"] = dataset_id
    payload["schema_json"] = schema_dict
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    data = _execute_data(
        client.table("educational_schemas").upsert(payload, on_conflict="dataset_id"),
        default=None,
    )
    return _single(data)


def get_educational_schema(dataset_id):
    client = get_supabase_client()
    if client is None or not dataset_id:
        return None

    data = _execute_data(
        client.table("educational_schemas").select("*").eq("dataset_id", dataset_id).limit(1),
        default=None,
    )
    return _single(data)


def save_analysis_result(dataset_id, result_type, result_json):
    client = get_supabase_client()
    if client is None or not dataset_id or not result_type:
        return None

    payload = {
        "dataset_id": dataset_id,
        "result_type": result_type,
        "result_json": result_json or {},
    }
    data = _execute_data(client.table("analysis_results").insert(payload), default=None)
    return _single(data)


def get_analysis_results(dataset_id, result_type=None):
    client = get_supabase_client()
    if client is None or not dataset_id:
        return []

    query = client.table("analysis_results").select("*").eq("dataset_id", dataset_id).order("created_at", desc=True)
    if result_type:
        query = query.eq("result_type", result_type)
    return _execute_data(query, default=[]) or []


def save_report(dataset_id, report_title, report_markdown, report_path=None):
    client = get_supabase_client()
    if client is None or not dataset_id or not report_title:
        return None

    payload = {
        "dataset_id": dataset_id,
        "report_title": report_title,
        "report_markdown": report_markdown,
        "report_path": report_path,
    }
    data = _execute_data(client.table("reports").insert(payload), default=None)
    return _single(data)


def get_reports(dataset_id):
    client = get_supabase_client()
    if client is None or not dataset_id:
        return []

    return _execute_data(
        client.table("reports").select("*").eq("dataset_id", dataset_id).order("created_at", desc=True),
        default=[],
    ) or []
