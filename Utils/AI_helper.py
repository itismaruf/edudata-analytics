import os
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# === Конфигурация ===
load_dotenv()
# Используем ключ из secrets или env.
# Если нет ключа — можно использовать Free Tier OpenRouter или аналоги, если пользователь предоставит.
# Здесь оставляем логику чтения ключа.
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# === Системный промпт (Persona) ===
SYSTEM_PROMPT = (
    "Ты — интеллектуальный аналитический помощник EduStat AI. "
    "Твоя цель — помогать пользователям (сотрудникам сферы образования) анализировать данные просто и понятно. "
    "Правила твоего поведения:"
    "1. Отвечай на русском языке. Твой тон — дружелюбный, профессиональный, спокойный."
    "2. Избегай слишком длинных лекций, но и не отвечай односложно. Ответ должен быть 'плавным' и понятным."
    "3. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО ПИСАТЬ КОД. Если тебя просят что-то сделать (построить график, посчитать), "
    "объясни словами, как это сделать в интерфейсе этого приложения, или просто опиши логику анализа."
    "4. Фокусируйся на выводах и инсайтах. Объясняй статистику простым языком."
    "5. Если данных нет или вопрос не ясен, вежливо уточни."
)

# === Управление сессией ===
def _get_history():
    """Получает историю чата из session_state."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return st.session_state["chat_history"]

def reset_ai_memory():
    """Полный сброс памяти ИИ."""
    st.session_state["chat_history"] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Также можно очищать контекстные переменные, если они хранятся отдельно

def _call_ai_api(messages, model="meta-llama/llama-3.3-70b-instruct:free"):
    """Внутренняя функция запроса к API."""
    if not API_KEY:
        return "❌ API-ключ не найден. Проверьте настройки (.env или secrets)."

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                 "HTTP-Referer": "http://localhost:8501", # OpenRouter requirement
                 "X-Title": "EduStat AI"
            },
            json={
                "model": model,
                "messages": messages
            },
            timeout=30
        )
        
        if resp.status_code != 200:
            return f"❌ Ошибка API ({resp.status_code}): {resp.text}"

        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        
        return f"❌ Пустой или некорректный ответ API: {data}"

    except Exception as e:
        return f"❌ Ошибка соединения: {e}"

# === Публичные функции ===

def send_user_message(user_text: str, model="meta-llama/llama-3.3-70b-instruct:free") -> str:
    """
    Отправляет сообщение пользователя в единый чат и возвращает ответ.
    Используется в основном окне чата.
    """
    if not user_text or not isinstance(user_text, str):
        return ""

    history = _get_history()
    # Добавляем сообщение пользователя, если его там еще нет 
    # (в чате UI оно обычно уже добавлено, но проверим)
    if not history or history[-1]["content"] != user_text:
         history.append({"role": "user", "content": user_text})
    
    # Запрос
    response_text = _call_ai_api(history, model)
    
    # Сохраняем ответ
    history.append({"role": "assistant", "content": response_text})
    
    return response_text


def connect_ai_context(df) -> None:
    """
    Тихо подключает контекст данных к истории чата, не вызывая API.
    Создает системное сообщение о структуре данных и имитирует подтверждение от ассистента.
    """
    # 1. Формируем описание датасета (переиспользуем логику)
    info_lines = [f"Датасет: {df.shape[0]} строк, {df.shape[1]} столбцов."]
    
    # Упрощенное описание колонок
    for col in df.columns[:50]:
        dtype = str(df[col].dtype)
        if hasattr(df[col], 'dt'): # datetime
             info_lines.append(f"- {col} (Date): {df[col].min()} ... {df[col].max()}")
        elif pd.api.types.is_numeric_dtype(df[col]):
             desc = df[col].describe()
             info_lines.append(f"- {col} (Num): range=[{desc.get('min',0):.2f}, {desc.get('max',0):.2f}], mean={desc.get('mean',0):.2f}")
        else:
             unique_str = ", ".join(map(str, df[col].dropna().unique()[:3]))
             info_lines.append(f"- {col} (Cat): {df[col].nunique()} уникальных, примеры: [{unique_str}]")
             
    if len(df.columns) > 50:
        info_lines.append("... (остальные колонки скрыты для краткости)")

    dataset_summary = "\n".join(info_lines)
    
    msg_content = (
        f"Пользователь загрузил новые данные. Структура:\n{dataset_summary}\n\n"
        "Запомни этот контекст для будущих вопросов."
    )
    
    # 2. Обновляем историю без вызова API
    history = _get_history()
    
    # Добавляем контекст как сообщение пользователя (или системы, но user надежнее для моделей)
    history.append({"role": "user", "content": msg_content})
    
    # Добавляем фейковый ответ ассистента
    history.append({"role": "assistant", "content": "Контекст данных принят. Я готов отвечать на вопросы по этому датасету."})

    
def notify_ai_about_context(df, user_goal="", model="meta-llama/llama-3.3-70b-instruct:free") -> str:
    """
    DEPRECATED: Используйте connect_ai_context для тихого подключения.
    Оставлено для совместимости, если где-то еще вызывается.
    """
    # Если нужно, можно перенаправить на connect_ai_context, но тут возврат str, так что оставим пока
    return "Функция устарела. Используйте кнопку 'Подключить ИИ' в новом интерфейсе."


def notify_ai_about_correlation(df, model="meta-llama/llama-3.3-70b-instruct:free") -> str:
    """Фиксирует найденные корреляции в истории чата."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return "Недостаточно данных для корреляции."

    corr_matrix = numeric_df.corr().abs()
    # Берем верхний треугольник без диагонали
    import numpy as np
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_stacked = corr_matrix.where(mask).stack().sort_values(ascending=False)
    
    top_corrs = corr_stacked.head(5)
    
    text_report = "Я провел корреляционный анализ. Топ-5 связей:\n"
    for (col1, col2), val in top_corrs.items():
        text_report += f"- {col1} <-> {col2}: {val:.2f}\n"
        
    text_report += "\nПрокомментируй эти связи."

    history = _get_history()
    history.append({"role": "user", "content": text_report})
    
    resp = _call_ai_api(history, model)
    history.append({"role": "assistant", "content": resp})
    
    return resp


def connect_ai_pivot(pivot_table, index_cols_str, value_col, agg_func) -> None:
    """
    Тихо подключает контекст сводной таблицы к истории чата.
    """
    # Преобразуем таблицу в текст (Markdown)
    table_md = pivot_table.head(10).to_markdown(index=False)
    
    msg = (
        f"Я построил сводную таблицу.\n"
        f"Группировка по: {index_cols_str}\n"
        f"Агрегация: {agg_func} от {value_col}\n"
        f"Вот первые 10 строк результата:\n{table_md}\n\n"
        "Запомни этот контекст."
    )
    
    history = _get_history()
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": "Контекст сводной таблицы принят."})


def notify_ai_about_pivot(pivot_table, index_cols_str, value_col, agg_func, model="meta-llama/llama-3.3-70b-instruct:free") -> str:
    """DEPRECATED: Use connect_ai_pivot instead."""
    return "Функция устарела."

# --- Alias для совместимости (если нужно, но лучше обновить вызовы) ---
update_context = None # Больше не используется, контекст теперь в истории
get_chatgpt_response = None # Удалено