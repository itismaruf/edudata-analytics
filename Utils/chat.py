import streamlit as st

from Utils.AI_helper import send_user_message, reset_ai_memory

def continue_chat(user_message):
    """Обрабатывает сообщение пользователя с учётом контекста проекта."""
    if not user_message or not isinstance(user_message, str):
        return "❌ Пустой или некорректный запрос."

    return send_user_message(user_message.strip())


def render_message(text: str, sender: str):
    # Map 'ai' to 'assistant' for Streamlit's native component
    role = "user" if sender == "user" else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)


def reset_chat_history():
    """
    Очищает историю чата в session_state.
    """
    reset_ai_memory()