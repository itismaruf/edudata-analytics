import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional, Tuple, Dict
import time

# Интеграция с ИИ
from Utils.AI_helper import (
    notify_ai_about_correlation,
    connect_ai_pivot,
    send_user_message
)

# === Внутренние зависимости ===
def apply_numeric_filters(
    df: pd.DataFrame,
    numeric_filters: Optional[Dict[str, Tuple[float, float]]]
) -> pd.DataFrame:
    """Фильтрует числовые колонки по диапазонам."""
    if not numeric_filters:
        return df
    try:
        for col, (min_val, max_val) in numeric_filters.items():
            if col not in df or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if min_val != max_val:
                df = df[df[col].between(min_val, max_val)]
    except Exception as e:
        st.warning(f"Ошибка при применении фильтров: {e}")
    return df

def is_temporal(column_name: str, series: pd.Series) -> bool:
    """Определяет, является ли колонка временной."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    keywords = ["date", "time", "year", "month"]
    return any(key in column_name.lower() for key in keywords)

# === Авто- и ручная визуализация ===
def generate_manual_chart(
    df: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    z: Optional[str] = None,
    chart_type: str = "Гистограмма"
):
    """Строит график выбранного типа."""
    try:
        if x not in df.columns:
            return None
        
        # Сортировка для Line plot
        if chart_type == "Лайнплот":
            df = df.sort_values(by=x)

        if chart_type == "Гистограмма":
            return px.histogram(df, x=x)
        elif chart_type == "Круговая диаграмма":
            counts = df[x].value_counts()
            return px.pie(names=counts.index, values=counts.values)
        elif chart_type == "Точечный график" and y:
            return px.scatter(df, x=x, y=y)
        elif chart_type == "Boxplot" and y:
            return px.box(df, x=x, y=y)
        elif chart_type == "Bar-график" and y:
            return px.bar(df, x=x, y=y)
        elif chart_type == "Лайнплот" and y:
            return px.line(df, x=x, y=y)
        elif chart_type == "3D Точечный" and y and z:
            return px.scatter_3d(df, x=x, y=y, z=z, color=y)
            
    except Exception as e:
        st.warning(f"Ошибка при построении графика ({chart_type}): {e}")
    return None

def generate_auto_chart(
    df: pd.DataFrame,
    x: str,
    y: Optional[str] = None
):
    """Автоматически подбирает тип графика."""
    try:
        if x not in df.columns: return None
        x_num = pd.api.types.is_numeric_dtype(df[x])
        y_num = pd.api.types.is_numeric_dtype(df[y]) if y else False
        is_time_x = is_temporal(x, df[x])

        if y:
            if x_num and y_num:
                if is_time_x:
                     return px.line(df.sort_values(by=x), x=x, y=y)
                return px.scatter(df, x=x, y=y)
            if not x_num and y_num:
                return px.bar(df.groupby(x)[y].mean().reset_index(), x=x, y=y)
            return px.bar(df, x=x, y=y)
        else:
            if x_num:
                return px.histogram(df, x=x)
            counts = df[x].value_counts()
            return px.pie(names=counts.index, values=counts.values)
    except Exception as e:
        st.warning(f"Ошибка авто-визуализации: {e}")
    return None

def plot_data_visualizations(df, x, y=None, z=None, numeric_filters=None, chart_type="Автоматически"):
    if x not in df.columns:
        st.warning("Некорректный выбор X.")
        return None

    df_filtered = apply_numeric_filters(df, numeric_filters or {})

    if df_filtered.empty:
        st.info("После фильтрации данных не осталось.")
        return None

    fig = None
    if chart_type == "Автоматически":
        fig = generate_auto_chart(df_filtered, x, y)
    else:
        fig = generate_manual_chart(df_filtered, x, y, z, chart_type)

    return fig


# === Рекомендации от ИИ ===
def suggest_visualization_combinations(df_info: str) -> str:
    """Запрашивает у ИИ рекомендации по визуализациям."""
    try:
        prompt = (
            f"Вот переменные: {df_info}. "
            "Предложи 2–3 интересные комбинации для визуализации (X и Y) "
            "и коротко поясни, что можно увидеть."
        )
        return send_user_message(prompt)
    except Exception as e:
        return f"Не удалось получить рекомендации: {e}"

# === Корреляции ===
def plot_correlation_heatmap(df: pd.DataFrame):
    """Строит тепловую карту корреляций."""
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr().round(2)
    return px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="🔗 Тепловая карта корреляций"
    )

# === Pivot ===
def generate_pivot_table(df: pd.DataFrame, index_cols: list, value_col: str, agg_func: str = "mean"):
    """Строит сводную таблицу по списку index_cols с агрегированием value_col."""
    if not index_cols or value_col not in df.columns:
        return None

    if agg_func not in {"mean", "sum", "count"}:
        return None

    grouped = df.groupby(index_cols, as_index=False)[value_col]

    if agg_func == "mean":
        pivot = grouped.mean()
    elif agg_func == "sum":
        pivot = grouped.sum()
    elif agg_func == "count":
        pivot = grouped.count()

    # Имя агрегированной колонки
    agg_col_name = f"{agg_func}({value_col})"
    # Последняя колонка - это результат агрегации, переименуем её
    pivot = pivot.rename(columns={value_col: agg_col_name})
    
    # Сортировка по значению
    pivot = pivot.sort_values(by=agg_col_name, ascending=False).reset_index(drop=True)
    
    return pivot


# === UI Components ===

def show_chart_tab(df: pd.DataFrame) -> None:
    """Вкладка: выбор переменных, тип графика, фильтры и построение графика."""
    st.subheader("🧭 Выбор переменных")

    def get_safe_index(key: str, options: list, default: int = 0) -> int:
        idx = st.session_state.get(key, default)
        if idx < 0 or idx >= len(options):
            return 0
        return idx

    # X и Y в одной строке
    col1, col2 = st.columns(2)
    with col1:
        x_idx = get_safe_index("eda_x_index", df.columns)
        x = st.selectbox(
            "🟥 Ось X",
            df.columns,
            index=x_idx,
            key="eda_x",
        )
    with col2:
        y_options = ["— не выбрано —"] + list(df.columns)
        y_idx = get_safe_index("eda_y_index", y_options)
        y = st.selectbox(
            "🟦 Ось Y (необязательно)",
            y_options,
            index=y_idx,
            key="eda_y",
        )
        if y == "— не выбрано —":
            y = None

    # Синхронизация состояний
    try:
        if x in df.columns:
            st.session_state["eda_x_index"] = list(df.columns).index(x)
        if y is None:
            st.session_state["eda_y_index"] = 0
        elif y in df.columns:
            st.session_state["eda_y_index"] = list(df.columns).index(y) + 1
    except ValueError:
        pass 

    if x == y and y is not None:
        st.warning("Переменные X и Y не должны совпадать.")
        y = None

    st.markdown("---")

    chart_help = {
        "Автоматически": "Система сама подберет лучший график.",
        "Гистограмма": "Показывает распределение одной переменной (например, баллы учеников).",
        "Круговая диаграмма": "Показывает доли категорий (например, % школ по регионам).",
        "Точечный график": "Показывает связь двух переменных (например, баллы vs посещаемость).",
        "Boxplot": "Сравнивает распределения по группам (медиана, квартили).",
        "Bar-график": "Сравнение значений по категориям.",
        "Лайнплот": "Идеально для динамики во времени (например, успеваемость по годам).",
        "3D Точечный": "Показывает связь ТРЕХ переменных в 3D пространстве."
    }

    with st.expander("🎨 Тип графика", expanded=True):
        chart_options = [
            "Автоматически",
            "Гистограмма",
            "Круговая диаграмма",
            "Точечный график",
            "Boxplot",
            "Bar-график",
            "Лайнплот",
            "3D Точечный"
        ]
        chart_idx = get_safe_index("eda_chart_index", chart_options)
        chart_type = st.selectbox(
            "Выберите подходящий тип графика",
            options=chart_options,
            index=chart_idx,
            key="eda_chart",
        )
        st.session_state["eda_chart_index"] = chart_options.index(chart_type)
        
        if chart_type in chart_help:
            st.caption(f"ℹ️ {chart_help[chart_type]}")
        
    st.markdown("---")
    
    # 3D selection
    z = None
    if "3D" in chart_type:
        st.info("Для 3D графика выберите третью переменную (Z).")
        col_z1, col_z2 = st.columns(2)
        with col_z1:
            z_options = [c for c in df.columns if c not in [x, y]]
            z_idx = get_safe_index("eda_z_index", z_options)
            z = st.selectbox("🟩 Ось Z", z_options, index=z_idx, key="eda_z")
            if z in z_options: 
                try: 
                    st.session_state["eda_z_index"] = z_options.index(z)
                except: pass

    # График с фильтрами
    with st.expander("📈 График с фильтрами", expanded=True):
        filters = {}
        cols_to_filter = [x] + ([y] if y else []) + ([z] if z else [])
        for col in dict.fromkeys(cols_to_filter):
            if col and pd.api.types.is_numeric_dtype(df[col]):
                lo, hi = float(df[col].min()), float(df[col].max())
                if lo != hi:
                    sel = st.slider(
                        f"Фильтр по {col}",
                        min_value=lo,
                        max_value=hi,
                        value=st.session_state.get(f"slider_{col}", (lo, hi)),
                        key=f"slider_{col}",
                    )
                    filters[col] = sel

        if st.button("📊 Построить график", key="build_chart_btn"):
            with st.spinner("Построение графика..."):
                time.sleep(1.0)
                fig = plot_data_visualizations(
                    df=df,
                    x=x,
                    y=y,
                    z=z,
                    numeric_filters=filters,
                    chart_type=chart_type,
                )
                st.session_state["eda_fig"] = fig
                st.session_state["eda_built"] = True
        
        if st.session_state.get("eda_built"):
            fig = st.session_state.get("eda_fig")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Не удалось построить график. Проверьте данные.")
        else:
             st.info("🎯 Нажмите «Построить график» для визуализации.")            


def show_ai_suggestions(df: pd.DataFrame) -> None:
    """Блок с советами от ИИ по визуализациям (вынесен отдельно)."""
    with st.expander("💡 Получить советы для визуализации от ИИ"):
        if st.button("✨ Предложи комбинации", key="suggest_combinations"):
            df_info = f"Переменные: {', '.join(df.columns)}"
            with st.spinner("Генерируем рекомендации..."):
                time.sleep(2)
                st.session_state["eda_suggestion"] = suggest_visualization_combinations(df_info)

        if "eda_suggestion" in st.session_state:
            st.markdown("**📝 Рекомендации от ИИ:**")
            st.info(st.session_state["eda_suggestion"], icon="🤖")


def show_correlation_tab(df: pd.DataFrame) -> None:
    """Вкладка: тепловая карта корреляций."""
    st.subheader("❄️ Корреляционный анализ")
    st.caption("Показывает силу связи между числовыми переменными. 1 = сильная связь, 0 = нет связи.")
    
    fig = plot_correlation_heatmap(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        # Добавляем кнопку отправки в ИИ
        if st.button("📤 Отправить корреляции в ИИ", key="send_corr_ai"):
             with st.spinner("Анализируем корреляции..."):
                 resp = notify_ai_about_correlation(df)
                 st.success("✅ Отправлено в ИИ")
                 st.info(resp)
    else:
        st.info("Недостаточно числовых данных для корреляции.")


def show_pivot_tab(df: pd.DataFrame) -> None:
    """Вкладка: сводные таблицы (pivot) и фиксация результата в ИИ + визуализация."""

    col1, col2 = st.columns(2)
    with col1:
        # Multiselect for grouping
        index_cols = st.multiselect(
            "Группировать по (можно выбрать несколько)",
            df.columns,
            default=[df.columns[0]] if len(df.columns) > 0 else None,
            key="pivot_index_cols"
        )

    with col2:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) == 0:
            st.warning("Нет числовых столбцов для агрегации.")
            return
        
        idx_val = st.session_state.get("pivot_value_index", 0)
        idx_val = idx_val if idx_val < len(num_cols) else 0

        value_col = st.selectbox(
            "Агрегировать",
            num_cols,
            index=idx_val,
            key="pivot_value",
        )
        st.session_state["pivot_value_index"] = list(num_cols).index(value_col)

    agg_options = ["mean", "sum", "count"]
    
    idx_agg = st.session_state.get("pivot_agg_index", 0)
    idx_agg = idx_agg if idx_agg < len(agg_options) else 0
    
    agg_func = st.radio(
        "Метод агрегации",
        agg_options,
        index=idx_agg,
        horizontal=True,
        key="pivot_agg",
    )
    st.session_state["pivot_agg_index"] = agg_options.index(agg_func)

    pivot_table = generate_pivot_table(df, index_cols, value_col, agg_func)
    
    if pivot_table is not None and not pivot_table.empty:
        st.dataframe(pivot_table, use_container_width=True)

        # === Кнопка фиксации в ИИ ===
        st.caption("Нажмите «Зафиксировать в ИИ», чтобы сохранить результат в чат.")
        if st.button("📤 Зафиксировать в ИИ", key="fix_pivot"):
            try:
                idx_str = ", ".join(index_cols)
                # Используем новую функцию тихого подключения
                from Utils.AI_helper import connect_ai_pivot
                connect_ai_pivot(pivot_table, idx_str, value_col, agg_func)
                st.session_state["pivot_saved"] = True
                st.success("✅ Сводная таблица зафиксирована в ИИ.")
            except Exception as e:
                st.error(f"Не удалось отправить сводную таблицу в ИИ: {e}")
        elif st.session_state.get("pivot_saved"):
            st.info("✅ Сводная таблица уже была зафиксирована.")

        st.markdown("---") 

        # === Визуализация сводной таблицы ===
        st.markdown("### 📉 Визуализация сводной таблицы")
        chart_type = st.selectbox(
            "Выберите тип графика",
            ["Bar", "Pie", "Line", "Sunburst (Иерархия)"],
            key="pivot_chart_type"
        )

        if st.button("Визуализировать", key="pivot_visualize"):
            with st.spinner("⏳ Строим визуализацию..."):
                time.sleep(1.0)

            try:
                # Сбрасываем индекс, чтобы index_cols стали обычными колонками
                pivot_to_plot = pivot_table.reset_index()
                agg_col = pivot_to_plot.columns[-1]  # Последняя колонка - значение агрегации
                
                # Если выбрано несколько колонок группировки, создаем комбинированную колонку для X оси
                if len(index_cols) > 1 and chart_type != "Sunburst (Иерархия)":
                    x_col = " | ".join(index_cols)
                    pivot_to_plot[x_col] = pivot_to_plot[index_cols].astype(str).agg(' - '.join, axis=1)
                    plot_x = x_col
                else:
                    plot_x = index_cols[0] if index_cols else "Index"

                if chart_type == "Bar":
                    fig = px.bar(
                        pivot_to_plot,
                        x=plot_x,
                        y=agg_col,
                        text=agg_col,
                        title=f"Bar график: {agg_func}({value_col})"
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Pie":
                    if pivot_to_plot[plot_x].nunique() >= 15:
                        st.warning("⚠️ Слишком много категорий для круговой диаграммы (более 15).")
                    else:
                        fig = px.pie(
                            pivot_to_plot,
                            names=plot_x,
                            values=agg_col,
                            title=f"Pie график: {agg_func}({value_col})"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Line":
                    fig = px.line(
                        pivot_to_plot,
                        x=plot_x,
                        y=agg_col,
                        markers=True,
                        title=f"Line график: {agg_func}({value_col})"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif chart_type == "Sunburst (Иерархия)":
                    if len(index_cols) < 2:
                        st.warning("⚠️ Для Sunburst выберите хотя бы 2 колонки группировки.")
                    else:
                        fig = px.sunburst(
                            pivot_to_plot,
                            path=index_cols,
                            values=agg_col,
                            title=f"Иерархия: {', '.join(index_cols)}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"⚠️ Не удалось построить график. Попробуйте выбрать другие параметры.")
                st.caption(f"Техническая информация: {str(e)[:100]}")

    else:
        st.info("⚠️ Выберите колонки для группировки.")