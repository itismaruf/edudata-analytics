import streamlit as st
from Utils.ui_config import show_splash, init_api_key, init_session
from Utils.i18n import init_language, install_streamlit_i18n, render_language_selector, t

# --- Config (Must be first) ---
st.set_page_config(page_title="EduStat AI", layout="wide")
init_language()
install_streamlit_i18n()

# --- Init Utils ---
show_splash()
render_language_selector()
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
    t("Главная"): [
        st.Page(show_home, title=t("Главная"), icon="🏠", default=True),
    ],
    t("Данные"): [
        st.Page("pages/00_Projects.py", title=t("Проекты"), icon="🗂️"),
        st.Page("pages/01_Data_Upload.py", title=t("Загрузка данных"), icon="📥"),
        st.Page("pages/02_Auto_Processing.py", title=t("Автообработка данных"), icon="🛡️"),
        st.Page("pages/03_Missing_Values.py", title=t("Обработка пропусков"), icon="⚙️"),
        st.Page("pages/04_Outlier_Handling.py", title=t("Обработка выбросов"), icon="🚩"),
    ],
    t("Анализ"): [
        st.Page("pages/05_Visual_Analysis.py", title=t("Визуальный анализ"), icon="📊"),
        st.Page("pages/06_Pivot_Tables.py", title=t("Сводные таблицы"), icon="📟"),
        st.Page("pages/07_Group_Comparison.py", title=t("Сравнение групп"), icon="⚖️"),
    ],
    t("Моделирование"): [
        st.Page("pages/08_Logistic_Regression.py", title=t("Логистическая регрессия"), icon="📈"),
        st.Page("pages/09_CatBoost_Modeling.py", title=t("CatBoost моделирование"), icon="🐈"),
    ],
    t("Помощь"): [
        st.Page("pages/10_AI_Chat.py", title=t("Чат с ИИ"), icon="💬"),
    ]
}

pg = st.navigation(pages)
pg.run()
