import streamlit as st
from Utils.ui_config import show_splash, init_api_key, init_session

# --- Config (Must be first) ---
st.set_page_config(page_title="EduStat AI", layout="wide")

# --- Init Utils ---
show_splash()
init_api_key()
init_session()

# --- Home Page Content ---
def show_home():
    st.title("👋 EduStat AI")
    st.caption("Платформа для интеллектуального анализа и моделирования данных")
    st.markdown("---")

    # --- Карточки разделов ---
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container():
            st.success("📥 **Загрузка данных**\n\nНачните работу с загрузки вашего датасета (CSV, Excel).")
        with st.container():
            st.info("📊 **Визуализация**\n\nСтройте графики, ищите инсайты и получайте советы от ИИ.")

    with col2:
        with st.container():
            st.warning("🛡️ **Автообработка**\n\nБыстрая очистка данных от пропусков и выбросов в один клик.")
        with st.container():
            st.error("📈 **Моделирование**\n\nОбучайте модели (LogReg, CatBoost) и делайте прогнозы.")

    with col3:
        with st.container():
            st.info("⚙️ **Ручная очистка**\n\nТонкая настройка удаления пропусков и аномалий.")
        with st.container():
            st.success("⚖️ **Сравнение групп**\n\nПроверяйте статистические гипотезы (T-test, ANOVA).")



# --- Navigation Setup ---
pages = {
    "Главная": [
        st.Page(show_home, title="Главная", icon="🏠", default=True),
    ],
    "Данные": [
        st.Page("pages/00_Projects.py", title="Проекты", icon="🗂️"),
        st.Page("pages/01_Data_Upload.py", title="Загрузка данных", icon="📥"),
        st.Page("pages/02_Auto_Processing.py", title="Автообработка данных", icon="🛡️"),
        st.Page("pages/03_Missing_Values.py", title="Обработка пропусков", icon="⚙️"),
        st.Page("pages/04_Outlier_Handling.py", title="Обработка выбросов", icon="🚩"),
    ],
    "Анализ": [
        st.Page("pages/05_Visual_Analysis.py", title="Визуальный анализ", icon="📊"),
        st.Page("pages/06_Pivot_Tables.py", title="Сводные таблицы", icon="📟"),
        st.Page("pages/07_Group_Comparison.py", title="Сравнение групп", icon="⚖️"),
    ],
    "Моделирование": [
        st.Page("pages/08_Logistic_Regression.py", title="Логистическая регрессия", icon="📈"),
        st.Page("pages/09_CatBoost_Modeling.py", title="CatBoost моделирование", icon="🐈"),
    ],
    "Помощь": [
        st.Page("pages/10_AI_Chat.py", title="Чат с ИИ", icon="💬"),
    ]
}

pg = st.navigation(pages)
pg.run()
