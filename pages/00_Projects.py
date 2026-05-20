import io
from datetime import datetime

import streamlit as st

from Utils.database_utils import (
    create_project,
    delete_project,
    download_dataset_file,
    get_datasets,
    get_projects,
)
from Utils.project_state import (
    get_latest_project_snapshot,
    restore_project_snapshot,
    save_current_project_state,
)
from Utils.supabase_client import is_supabase_configured
from Utils.upload_utils import load_data


DEFAULT_USER_ID = "default_user"


class NamedBytesIO(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def format_date(value):
    if not value:
        return "Не указано"
    try:
        normalized = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return str(value)


def project_description(project):
    return project.get("description") or "Описание не добавлено"


def visible_datasets(datasets):
    return [
        item for item in datasets
        if not str(item.get("original_filename") or "").startswith("project_snapshot_")
    ]


def open_project(project):
    st.session_state["current_project_id"] = project["id"]
    st.session_state["current_project_name"] = project.get("name")
    latest_snapshot = get_latest_project_snapshot(project["id"])

    if latest_snapshot:
        with st.spinner("Восстанавливаем последний сохраненный снимок..."):
            restored, error = restore_project_snapshot(project["id"], project.get("name"))
        if error:
            st.warning(error)
        else:
            st.success("Проект открыт и восстановлен.")
    else:
        st.success("Проект открыт.")
    st.rerun()


def reset_current_project_if_needed(project_id):
    if st.session_state.get("current_project_id") != project_id:
        return

    for key in (
        "current_project_id",
        "current_project_name",
        "current_dataset_id",
        "df",
        "original_filename",
        "pending_dataset_file_bytes",
        "pending_dataset_filename",
    ):
        st.session_state.pop(key, None)


def render_dataset_loader(current_project_id, datasets):
    with st.expander("📦 Сохраненные датасеты проекта", expanded=False):
        if not current_project_id:
            st.info("Откройте проект, чтобы загрузить сохраненные датасеты.")
            return

        if not datasets:
            st.info("В этом проекте пока нет сохраненных датасетов.")
            return

        dataset_options = {
            (
                f"{item.get('original_filename')} | "
                f"{item.get('rows_count')} строк x {item.get('columns_count')} столбцов | "
                f"{format_date(item.get('created_at'))}"
            ): item
            for item in datasets
        }
        selected_label = st.selectbox("Сохраненные датасеты", list(dataset_options.keys()))
        selected_dataset = dataset_options[selected_label]

        if st.button("📥 Загрузить датасет", use_container_width=True):
            file_bytes = download_dataset_file(selected_dataset.get("file_path"))
            if not file_bytes:
                st.error("Не удалось скачать датасет из Supabase Storage.")
                return

            try:
                file_obj = NamedBytesIO(file_bytes, selected_dataset["original_filename"])
                df = load_data(file_obj)
                st.session_state["df"] = df
                st.session_state["current_dataset_id"] = selected_dataset["id"]
                st.session_state["original_filename"] = selected_dataset["original_filename"]
                st.session_state.pop("pending_dataset_file_bytes", None)
                st.session_state.pop("pending_dataset_filename", None)
                st.success("Датасет загружен в текущую сессию.")
            except Exception as exc:
                st.error(f"Не удалось загрузить датасет: {exc}")


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
    }
    div[data-testid="stForm"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        background: #ffffff;
    }
    .project-card-active {
        padding: 0.2rem 0 0.45rem 0;
        border-top: 3px solid #2563eb;
    }
    .muted-text {
        color: #64748b;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📁 Проекты")
st.caption("Создавайте проекты, сохраняйте результаты анализа и продолжайте работу в любое время.")

if not is_supabase_configured():
    st.warning(
        "Supabase не настроен. Добавьте SUPABASE_URL и SUPABASE_KEY в Streamlit secrets или переменные окружения."
    )
    st.stop()

current_project_id = st.session_state.get("current_project_id")
current_project_name = st.session_state.get("current_project_name")

projects = get_projects(user_id=DEFAULT_USER_ID)
datasets_by_project = {
    project["id"]: visible_datasets(get_datasets(project["id"]))
    for project in projects
}
current_project = next((item for item in projects if item["id"] == current_project_id), None)
current_datasets = datasets_by_project.get(current_project_id, [])
total_datasets = sum(len(items) for items in datasets_by_project.values())

with st.sidebar:
    st.caption("Текущий проект")
    st.write(current_project_name or "Не выбран")

summary_cols = st.columns(3)
summary_cols[0].metric("Проектов", len(projects))
summary_cols[1].metric("Датасетов", total_datasets)
summary_cols[2].metric("Текущий проект", "Выбран" if current_project_id else "Нет")

st.divider()

if current_project:
    with st.container(border=True):
        st.markdown('<div class="project-card-active"></div>', unsafe_allow_html=True)
        top_left, top_right = st.columns([3, 1])
        with top_left:
            st.subheader(current_project.get("name") or "Без названия")
            st.caption(project_description(current_project))
        with top_right:
            st.metric("Датасетов", len(current_datasets))

        meta_cols = st.columns(3)
        meta_cols[0].caption(f"Создан: {format_date(current_project.get('created_at'))}")
        meta_cols[1].caption(f"ID: {str(current_project.get('id'))[:8]}...")
        meta_cols[2].caption("Состояние: активен")

        action_cols = st.columns(2)
        with action_cols[0]:
            save_disabled = "df" not in st.session_state
            if st.button("💾 Сохранить проект", disabled=save_disabled, use_container_width=True):
                with st.spinner("Сохраняем датасет, настройки и результаты анализа..."):
                    metadata, error = save_current_project_state(
                        current_project_id,
                        st.session_state.get("current_project_name"),
                    )
                if error:
                    st.error(error)
                else:
                    st.success(
                        f"Проект сохранен: {metadata['rows_count']} строк, {metadata['columns_count']} столбцов."
                    )
                    st.rerun()
            if save_disabled:
                st.caption("Загрузите датасет, чтобы сохранить проект.")

        with action_cols[1]:
            latest_snapshot = get_latest_project_snapshot(current_project_id)
            restore_disabled = latest_snapshot is None
            if st.button("↩️ Восстановить последний снимок", disabled=restore_disabled, use_container_width=True):
                with st.spinner("Восстанавливаем сохраненное состояние проекта..."):
                    restored, error = restore_project_snapshot(
                        current_project_id,
                        st.session_state.get("current_project_name"),
                    )
                if error:
                    st.error(error)
                else:
                    st.success("Последний снимок проекта восстановлен.")
                    st.rerun()
            if restore_disabled:
                st.caption("Сохраненных снимков пока нет.")
else:
    with st.container(border=True):
        st.info("Выберите проект из списка или создайте новый, чтобы начать сохранять результаты анализа.")

render_dataset_loader(current_project_id, current_datasets)

st.divider()

with st.expander("➕ Создать новый проект", expanded=not projects):
    with st.form("create_project_form", clear_on_submit=True):
        name_col, desc_col = st.columns([1, 2])
        with name_col:
            project_name = st.text_input("Название проекта", placeholder="Например: Анализ успеваемости")
        with desc_col:
            description = st.text_area(
                "Описание",
                placeholder="Коротко опишите цель проекта",
                height=86,
            )
        submitted = st.form_submit_button("➕ Создать проект", use_container_width=True)

    if submitted:
        if not project_name.strip():
            st.warning("Введите название проекта.")
        else:
            project = create_project(
                name=project_name.strip(),
                description=description.strip() or None,
                user_id=DEFAULT_USER_ID,
            )
            if project:
                st.success("Проект создан.")
                st.session_state["current_project_id"] = project["id"]
                st.session_state["current_project_name"] = project["name"]
                st.rerun()
            else:
                st.error("Не удалось создать проект. Проверьте таблицы Supabase и права доступа.")

st.divider()

st.subheader("Список проектов")

filter_col, sort_col, search_col = st.columns([1, 1, 2])
with filter_col:
    project_filter = st.selectbox("Фильтр", ["Все проекты", "Только текущий"])
with sort_col:
    sort_mode = st.selectbox("Сортировка", ["По дате создания", "По названию"])
with search_col:
    search_query = st.text_input("Поиск", placeholder="Введите название проекта")

visible_projects = list(projects)
if project_filter == "Только текущий":
    visible_projects = [item for item in visible_projects if item["id"] == current_project_id]

if search_query.strip():
    query = search_query.strip().lower()
    visible_projects = [
        item for item in visible_projects
        if query in (item.get("name") or "").lower()
    ]

if sort_mode == "По названию":
    visible_projects = sorted(visible_projects, key=lambda item: (item.get("name") or "").lower())
else:
    visible_projects = sorted(
        visible_projects,
        key=lambda item: item.get("created_at") or "",
        reverse=True,
    )

if not projects:
    with st.container(border=True):
        st.info("У вас пока нет проектов. Создайте первый проект, чтобы сохранять результаты анализа.")
elif not visible_projects:
    st.info("По текущим условиям ничего не найдено.")
else:
    for project in visible_projects:
        project_id = project["id"]
        is_current = project_id == current_project_id
        datasets_count = len(datasets_by_project.get(project_id, []))

        with st.container(border=True):
            info_col, metric_col, actions_col = st.columns([3, 1, 1.3])
            with info_col:
                title = project.get("name") or "Без названия"
                if is_current:
                    st.markdown(f"**📌 {title}**")
                    st.caption("Текущий проект")
                else:
                    st.markdown(f"**{title}**")
                st.caption(project_description(project))
                st.caption(f"Создан: {format_date(project.get('created_at'))}")

            with metric_col:
                st.metric("Датасетов", datasets_count)

            with actions_col:
                open_disabled = is_current
                if st.button("📂 Открыть", key=f"open_project_{project_id}", disabled=open_disabled, use_container_width=True):
                    open_project(project)

                confirm_key = f"confirm_delete_project_{project_id}"
                if st.button("🗑️ Удалить", key=f"delete_project_{project_id}", use_container_width=True):
                    st.session_state[confirm_key] = True

            if st.session_state.get(f"confirm_delete_project_{project_id}"):
                st.warning(f"Удалить проект «{project.get('name') or 'Без названия'}»? Это действие нельзя отменить.")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Да, удалить", key=f"confirm_delete_yes_{project_id}", use_container_width=True):
                        deleted = delete_project(project_id)
                        if deleted:
                            reset_current_project_if_needed(project_id)
                            st.session_state.pop(f"confirm_delete_project_{project_id}", None)
                            st.success("Проект удален.")
                            st.rerun()
                        else:
                            st.error("Не удалось удалить проект. Проверьте права доступа Supabase.")
                with cancel_col:
                    if st.button("Отмена", key=f"confirm_delete_no_{project_id}", use_container_width=True):
                        st.session_state.pop(f"confirm_delete_project_{project_id}", None)
                        st.rerun()
