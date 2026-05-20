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
def _read_secret_or_env(name, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name) or default


def _read_model_list(name, default_models):
    raw_value = _read_secret_or_env(name)
    if not raw_value:
        return list(default_models)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


API_KEY = _read_secret_or_env("OPENAI_API_KEY")
DEFAULT_LLM_MODEL = _read_secret_or_env(
    "OPENROUTER_MODEL",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
)
FALLBACK_LLM_MODELS = _read_model_list(
    "OPENROUTER_FALLBACK_MODELS",
    [
        "openrouter/free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ],
)

# === Системный промпт (Persona) ===
SYSTEM_PROMPT = (
    "Ты — интеллектуальный аналитический помощник EduStat AI. "
    "Твоя цель — помогать пользователям (сотрудникам сферы образования) анализировать данные просто и понятно. "
    
    "ВОЗМОЖНОСТИ ПРИЛОЖЕНИЯ (ты можешь подсказывать, какой раздел использовать):\n"
    "1. 📥 Загрузка данных — загрузка CSV/Excel файлов для анализа.\n"
    "2. 🛡️ Автообработка — автоматическая очистка пропусков и выбросов в один клик.\n"
    "3. ⚙️ Обработка пропусков — ручная настройка заполнения/удаления NaN значений.\n"
    "4. 🚩 Обработка выбросов — IQR, Z-score методы для удаления аномалий.\n"
    "5. 📊 Визуальный анализ — построение графиков (гистограммы, scatter, boxplot, 3D), корреляционный анализ.\n"
    "6. 📟 Сводные таблицы — группировка данных по категориям с агрегацией (сумма, среднее, количество).\n"
    "7. ⚖️ Сравнение групп — статистические тесты: t-test (2 группы), ANOVA (3+ групп), Chi-squared (категории).\n"
    "8. 📈 Логистическая регрессия — бинарная классификация, интерпретация коэффициентов, важность признаков.\n"
    "9. 🐈 CatBoost моделирование — мощная модель для классификации и регрессии, работает с категориальными признаками.\n"
    "10. 💬 Чат с ИИ (ты!) — помощь в интерпретации результатов, советы по анализу.\n\n"
    
    "ПРАВИЛА ПОВЕДЕНИЯ:\n"
    "1. Отвечай на русском языке. Тон — дружелюбный, профессиональный, спокойный.\n"
    "2. Избегай слишком длинных лекций, но и не отвечай односложно. Ответ должен быть 'плавным' и понятным.\n"
    "3. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО ПИСАТЬ КОД. Объясняй словами, как использовать интерфейс приложения.\n"
    "4. Фокусируйся на выводах и инсайтах. Объясняй статистику простым языком.\n"
    "5. Если пользователь спрашивает 'что делать' — подскажи подходящий раздел приложения.\n"
    "6. При вопросах о метриках (Recall, Precision, F1, AUC) объясняй для каких задач они важны:\n"
    "   - Recall важен, когда пропуск положительного случая критичен (выявление болезни, мошенничество).\n"
    "   - Precision важен, когда ложное срабатывание дорого (спам-фильтр, рекомендации).\n"
    "   - AUC показывает общее качество модели в различении классов.\n"
    "7. Если данных нет или вопрос не ясен, вежливо уточни.\n"
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


def _model_chain(primary_model=None):
    models = [primary_model or DEFAULT_LLM_MODEL] + FALLBACK_LLM_MODELS
    unique_models = []
    for item in models:
        if item and item not in unique_models:
            unique_models.append(item)
    return unique_models


def _api_error_details(resp):
    try:
        payload = resp.json()
    except Exception:
        return resp.text, None

    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error.get("message") or str(payload)
    metadata = error.get("metadata") or {}
    retry_after = metadata.get("retry_after_seconds") or metadata.get("retry_after_seconds_raw")
    return message, retry_after


def _should_try_next_model(status_code, error_message):
    if status_code == 429:
        return True
    if status_code == 404 and "No endpoints found" in str(error_message):
        return True
    return False


def _call_ai_api(messages, model=None):
    """Внутренняя функция запроса к API."""
    if not API_KEY:
        return "❌ API-ключ не найден. Проверьте настройки (.env или secrets)."

    last_error = None
    last_retry_after = None

    for selected_model in _model_chain(model):
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
                    "model": selected_model,
                    "messages": messages
                },
                timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
                return f"❌ Пустой или некорректный ответ API: {data}"

            error_message, retry_after = _api_error_details(resp)
            last_error = f"{selected_model}: {error_message}"
            last_retry_after = retry_after or last_retry_after

            if _should_try_next_model(resp.status_code, error_message):
                continue

            return f"❌ Ошибка API ({resp.status_code}): {error_message}"

        except Exception as e:
            last_error = f"{selected_model}: {e}"
            continue

    if last_retry_after:
        return (
            "❌ Сейчас LLM временно ограничен по лимитам OpenRouter. "
            f"Попробуйте еще раз примерно через {int(float(last_retry_after))} секунд."
        )

    return f"❌ Не удалось получить ответ от доступных LLM-моделей. Последняя ошибка: {last_error}"


# === Публичные функции ===

def send_user_message(user_text: str, model=None) -> str:
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

    
def notify_ai_about_context(df, user_goal="", model=None) -> str:
    """
    DEPRECATED: Используйте connect_ai_context для тихого подключения.
    Оставлено для совместимости, если где-то еще вызывается.
    """
    # Если нужно, можно перенаправить на connect_ai_context, но тут возврат str, так что оставим пока
    return "Функция устарела. Используйте кнопку 'Подключить ИИ' в новом интерфейсе."


def notify_ai_about_correlation(df, model=None) -> str:
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


def connect_ai_model_results(metrics: dict, model_type: str, target_col: str, top_features: list = None) -> None:
    """
    Тихо подключает результаты модели к истории чата.
    Позволяет ИИ объяснять метрики и давать советы.
    
    Args:
        metrics: словарь метрик {"Accuracy": 0.85, "Precision": 0.78, ...}
        model_type: тип модели ("Логистическая регрессия" или "CatBoost")
        target_col: название целевой переменной
        top_features: список важных признаков (опционально)
    """
    # Форматируем метрики
    metrics_text = "\n".join([f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in metrics.items()])
    
    msg = (
        f"Я обучил модель {model_type}.\n"
        f"Целевая переменная: {target_col}\n\n"
        f"Результаты модели:\n{metrics_text}\n"
    )
    
    if top_features:
        features_text = ", ".join(top_features[:5])
        msg += f"\nТоп-5 важных признаков: {features_text}\n"
    
    msg += "\nЗапомни эти результаты. Если я спрошу — объясни метрики простым языком."
    
    history = _get_history()
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": f"Результаты модели {model_type} приняты. Готов объяснить метрики или дать рекомендации."})


def notify_ai_about_pivot(pivot_table, index_cols_str, value_col, agg_func, model=None) -> str:
    """DEPRECATED: Use connect_ai_pivot instead."""
    return "Функция устарела."

# --- Alias для совместимости (если нужно, но лучше обновить вызовы) ---
update_context = None # Больше не используется, контекст теперь в истории
get_chatgpt_response = None # Удалено
