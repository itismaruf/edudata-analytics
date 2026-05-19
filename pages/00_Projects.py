import io

import streamlit as st

from Utils.database_utils import (
    create_project,
    download_dataset_file,
    get_datasets,
    get_projects,
)
from Utils.supabase_client import is_supabase_configured
from Utils.upload_utils import load_data


DEFAULT_USER_ID = "default_user"


class NamedBytesIO(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


st.title("Projects")

current_project_name = st.session_state.get("current_project_name")
if current_project_name:
    st.info(f"Current project: {current_project_name}")

if not is_supabase_configured():
    st.warning(
        "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets or environment variables."
    )
    st.stop()

with st.sidebar:
    if current_project_name:
        st.caption("Current project")
        st.write(current_project_name)

st.subheader("Create Project")
with st.form("create_project_form", clear_on_submit=True):
    project_name = st.text_input("Project name")
    description = st.text_area("Description")
    submitted = st.form_submit_button("Create Project")

if submitted:
    if not project_name.strip():
        st.warning("Project name is required.")
    else:
        project = create_project(
            name=project_name.strip(),
            description=description.strip() or None,
            user_id=DEFAULT_USER_ID,
        )
        if project:
            st.success("Project created.")
            st.session_state["current_project_id"] = project["id"]
            st.session_state["current_project_name"] = project["name"]
            st.rerun()
        else:
            st.error("Could not create project. Check Supabase tables and permissions.")

st.markdown("---")
st.subheader("Existing Projects")

projects = get_projects(user_id=DEFAULT_USER_ID)
if not projects:
    st.info("No projects yet.")
else:
    for project in projects:
        with st.container(border=True):
            col_info, col_action = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{project.get('name', 'Untitled')}**")
                if project.get("description"):
                    st.write(project["description"])
                st.caption(f"Created: {project.get('created_at', 'unknown')}")
            with col_action:
                if st.button("Open Project", key=f"open_project_{project['id']}"):
                    st.session_state["current_project_id"] = project["id"]
                    st.session_state["current_project_name"] = project.get("name")
                    st.success("Project opened.")
                    st.rerun()

st.markdown("---")
st.subheader("Load Saved Dataset")

current_project_id = st.session_state.get("current_project_id")
if not current_project_id:
    st.info("Open a project to load saved datasets.")
else:
    datasets = get_datasets(current_project_id)
    if not datasets:
        st.info("This project has no saved datasets yet.")
    else:
        dataset_options = {
            f"{item.get('original_filename')} | {item.get('rows_count')} rows x {item.get('columns_count')} cols | {item.get('created_at')}": item
            for item in datasets
        }
        selected_label = st.selectbox("Saved datasets", list(dataset_options.keys()))
        selected_dataset = dataset_options[selected_label]

        if st.button("Load Dataset"):
            file_bytes = download_dataset_file(selected_dataset.get("file_path"))
            if not file_bytes:
                st.error("Could not download dataset file from Supabase Storage.")
            else:
                try:
                    file_obj = NamedBytesIO(file_bytes, selected_dataset["original_filename"])
                    df = load_data(file_obj)
                    st.session_state["df"] = df
                    st.session_state["current_dataset_id"] = selected_dataset["id"]
                    st.session_state["original_filename"] = selected_dataset["original_filename"]
                    st.session_state.pop("pending_dataset_file_bytes", None)
                    st.session_state.pop("pending_dataset_filename", None)
                    st.success("Dataset loaded into the current session.")
                except Exception as exc:
                    st.error(f"Could not load dataset: {exc}")
