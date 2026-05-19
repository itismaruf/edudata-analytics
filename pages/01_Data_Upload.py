import streamlit as st
import pandas as pd
import io
from Utils.upload_utils import load_data, get_base_info, show_data_head, show_descriptive_stats, display_base_info
from Utils.AI_helper import connect_ai_context
from Utils.database_utils import create_dataset_record, upload_dataset_file
from Utils.supabase_client import is_supabase_configured


class NamedBytesIO(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def save_dataset_to_current_project(file_bytes, filename, df, project_id):
    file_obj = NamedBytesIO(file_bytes, filename)
    file_path = upload_dataset_file(file_obj, project_id)
    if not file_path:
        return None, "Could not upload dataset file. Check Supabase Storage bucket and permissions."

    dataset = create_dataset_record(
        project_id=project_id,
        original_filename=filename,
        file_path=file_path,
        rows_count=int(df.shape[0]),
        columns_count=int(df.shape[1]),
    )
    if not dataset:
        return None, "File uploaded, but dataset metadata could not be saved."
    return dataset, None


st.title("📥 Загрузка данных")
st.caption("Загрузите ваш файл в формате CSV или Excel для начала анализа.")

current_project_id = st.session_state.get("current_project_id")
current_project_name = st.session_state.get("current_project_name")

if current_project_name:
    st.info(f"Current project: {current_project_name}")
elif is_supabase_configured():
    st.info("You can analyze data without saving, or select/create a project to save it permanently.")

# --- Загрузка данных ---
if "df" not in st.session_state:
    uploaded_file = st.file_uploader(" ", type=["csv", "xlsx", "xls"])
    if not uploaded_file:
        st.info("⬆ Загрузите файл для анализа.", icon="📁")
    else:
        try:
            df = load_data(uploaded_file)
            st.session_state["df"] = df
            st.session_state["pending_dataset_file_bytes"] = uploaded_file.getvalue()
            st.session_state["pending_dataset_filename"] = uploaded_file.name
            st.session_state.pop("current_dataset_id", None)
            st.success("Данные успешно загружены", icon="✅")

            if is_supabase_configured() and current_project_id:
                with st.spinner("Saving dataset to current project..."):
                    dataset, save_error = save_dataset_to_current_project(
                        st.session_state["pending_dataset_file_bytes"],
                        st.session_state["pending_dataset_filename"],
                        df,
                        current_project_id,
                    )
                if dataset:
                    st.session_state["current_dataset_id"] = dataset["id"]
                    st.success("Dataset automatically saved to the current project.")
                else:
                    st.warning(save_error)
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

    if is_supabase_configured():
        st.markdown("---")
        st.subheader("💾 Save dataset")
        if current_project_id:
            if st.session_state.get("current_dataset_id"):
                st.success("Dataset is saved in the current project.")
            elif st.session_state.get("pending_dataset_file_bytes") and st.session_state.get("pending_dataset_filename"):
                st.caption("Automatic saving did not complete. You can retry saving this uploaded source file.")
                if st.button("Retry save dataset to project"):
                    with st.spinner("Saving dataset to project..."):
                        dataset, save_error = save_dataset_to_current_project(
                            st.session_state["pending_dataset_file_bytes"],
                            st.session_state["pending_dataset_filename"],
                            df,
                            current_project_id,
                        )
                        if dataset:
                            st.session_state["current_dataset_id"] = dataset["id"]
                            st.success("Dataset saved to project.")
                        else:
                            st.error(save_error)
            else:
                st.info("This dataset was loaded or created in session without an upload file to save.")
        else:
            st.info("You can analyze data without saving, or select/create a project to save it permanently.")

    # — Инициализация/обновление краткого summary —
    data_sig = (tuple(df.columns), df.shape)
    if st.session_state.get("_data_sig") != data_sig:
        summary = f"{df.shape[0]} строк, {df.shape[1]} столбцов; признаки: {', '.join(map(str, df.columns))}"
        st.session_state["_data_sig"] = data_sig
        st.session_state["data_summary"] = summary
        
        # update_context removed as state is now unified in chat history
    else:
        summary = st.session_state.get(
            "data_summary",
            f"{df.shape[0]} строк, {df.shape[1]} столбцов; признаки: {', '.join(map(str, df.columns))}"
        )

    st.markdown("---")
    # Блок подключения ИИ в экспандере
    st.subheader("🤖 Подключение ИИ")
    st.caption("Нажмите кнопку ниже, чтобы передать контекст данных ИИ. Это позволит ему отвечать на вопросы по вашему датасету.")

    if st.button("Подключить ИИ"):
        with st.spinner("Подключаем ИИ к вашим данным..."):
            import time
            time.sleep(2)
            connect_ai_context(df)
        st.success("✅ ИИ успешно подключен к данным! Теперь вы можете переходить в чат.")
