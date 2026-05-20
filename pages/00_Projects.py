import io

import streamlit as st

from Utils.database_utils import (
    create_project,
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


st.title("🗂️ Проекты")

current_project_name = st.session_state.get("current_project_name")
if current_project_name:
    st.info(f"Текущий проект: {current_project_name}")

if not is_supabase_configured():
    st.warning(
        "Supabase не настроен. Добавьте SUPABASE_URL и SUPABASE_KEY в Streamlit secrets или переменные окружения."
    )
    st.stop()

with st.sidebar:
    if current_project_name:
        st.caption("Текущий проект")
        st.write(current_project_name)

st.subheader("Создать проект")
with st.form("create_project_form", clear_on_submit=True):
    project_name = st.text_input("Название проекта")
    description = st.text_area("Описание")
    submitted = st.form_submit_button("Создать проект")

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

st.markdown("---")
st.subheader("Сохранение текущей работы")

current_project_id = st.session_state.get("current_project_id")
if not current_project_id:
    st.info("Откройте или создайте проект, чтобы сохранять работу.")
elif "df" not in st.session_state:
    st.info("Загрузите датасет, затем сохраните проект.")
else:
    col_save, col_restore = st.columns(2)
    with col_save:
        if st.button("💾 Сохранить проект", use_container_width=True):
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
    with col_restore:
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

st.markdown("---")
st.subheader("Список проектов")

projects = get_projects(user_id=DEFAULT_USER_ID)
if not projects:
    st.info("Проектов пока нет.")
else:
    for project in projects:
        with st.container(border=True):
            col_info, col_action = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{project.get('name', 'Без названия')}**")
                if project.get("description"):
                    st.write(project["description"])
                st.caption(f"Создан: {project.get('created_at', 'неизвестно')}")
            with col_action:
                if st.button("Открыть проект", key=f"open_project_{project['id']}"):
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

st.markdown("---")
st.subheader("Загрузить сохраненный датасет")

if not current_project_id:
    st.info("Откройте проект, чтобы загрузить сохраненные датасеты.")
else:
    datasets = get_datasets(current_project_id)
    if not datasets:
        st.info("В этом проекте пока нет сохраненных датасетов.")
    else:
        dataset_options = {
            f"{item.get('original_filename')} | {item.get('rows_count')} строк x {item.get('columns_count')} столбцов | {item.get('created_at')}": item
            for item in datasets
        }
        selected_label = st.selectbox("Сохраненные датасеты", list(dataset_options.keys()))
        selected_dataset = dataset_options[selected_label]

        if st.button("Загрузить датасет"):
            file_bytes = download_dataset_file(selected_dataset.get("file_path"))
            if not file_bytes:
                st.error("Не удалось скачать датасет из Supabase Storage.")
            else:
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
