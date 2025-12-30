import streamlit as st
import pandas as pd
from Utils.chat import continue_chat, render_message, reset_chat_history



st.title("💬 Поговорим о ваших данных?")
st.markdown("---")

# === CSS Styles ===
st.markdown("""
<style>
/* Общий контейнер сообщения */
.chat-row {
    display: flex;
    margin-bottom: 15px;
    width: 100%;
}

/* Сообщение пользователя - СПРАВА */
.chat-row.user {
    justify-content: flex-end;
}

.chat-row.user .bubble {
    background: linear-gradient(135deg, #007BFF, #0056b3);
    color: white;
    border-radius: 20px 20px 5px 20px;
    padding: 12px 18px;
    max-width: 75%;
    box-shadow: 0 2px 8px rgba(0,123,255,0.3);
}

/* Сообщение ИИ - СЛЕВА */
.chat-row.ai {
    justify-content: flex-start;
}

.chat-row.ai .bubble {
    background-color: #f0f2f5;
    color: #1a1a1a;
    border-radius: 20px 20px 20px 5px;
    padding: 12px 18px;
    max-width: 80%;
    border: 1px solid #e0e0e0;
}

/* Иконки/аватары */
.avatar {
    font-size: 1.5rem;
    margin: 0 8px;
    display: flex;
    align-items: flex-start;
}

/* Текст внутри bubble */
.bubble p {
    margin: 0;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

if st.button("🗑 Очистить чат"):
    reset_chat_history()
    st.success("Чат очищен.")

st.session_state.setdefault("chat_history", [])

# === Функция для рендера сообщения ===
def render_chat_message(content, role):
    if role == "user":
        st.markdown(f'''
        <div class="chat-row user">
            <div class="bubble">{content}</div>
            <div class="avatar">👤</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="chat-row ai">
            <div class="avatar">🤖</div>
            <div class="bubble">{content}</div>
        </div>
        ''', unsafe_allow_html=True)

# 1. Рендерим историю
for msg in st.session_state.chat_history:
    role = msg.get("role")
    if role == "system":
        continue
    content = msg.get("content") or msg.get("text", "")
    render_chat_message(content, role)

# 2. Ввод нового сообщения
if question := st.chat_input("Напишите свой вопрос…"):
    # Отображаем вопрос пользователя (Справа, синий)
    render_chat_message(question, "user")

    # Показываем загрузку и получаем ответ ИИ
    with st.spinner("⏳ ИИ думает..."):
        answer = continue_chat(question)
    
    # Отображаем ответ ИИ (Слева)
    render_chat_message(answer, "assistant")
