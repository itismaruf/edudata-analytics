import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from scipy import stats
from scipy.stats import gaussian_kde
import numpy as np

# ==== Утилиты ====
def is_numeric(series: pd.Series) -> bool:
    """Проверяет, является ли серия числовой."""
    return pd.api.types.is_numeric_dtype(series)

def is_categorical(series: pd.Series) -> bool:
    """Проверяет, является ли серия категориальной."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series)

# ==== Сводка по группам ====
def group_summary(df: pd.DataFrame, num_col: str, cat_col: str) -> pd.DataFrame:
    """Возвращает DataFrame со средними, SD, SE и количеством наблюдений по группам."""
    summary = (
        df.groupby(cat_col)[num_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={
            'mean': 'Среднее',
            'std': 'SD',
            'count': 'N',
            cat_col: 'Группа'
        })
    )
    summary['SE'] = summary['SD'] / summary['N']**0.5
    return summary

# ==== Вывод результатов теста ====
def display_test_result(test_name: str, stat_label: str, stat_value: float, p_value: float, alpha: float = 0.05):
    """Единый формат вывода результатов статистического теста."""
    if p_value < alpha:
        st.success(f"✅ {test_name}: различия значимы (p = {p_value:.4f})")
    else:
        st.info(f"ℹ️ {test_name}: различия незначимы (p = {p_value:.4f})")

    st.metric(label=stat_label, value=f"{stat_value:.4f}")
    st.metric(label="p‑value", value=f"{p_value:.4f}")

# ==== Bar chart со средними ====
def plot_group_means(summary_df: pd.DataFrame, title: str = "Сравнение средних значений"):
    """Строит bar chart со средними значениями и доверительными интервалами (SE)."""
    fig = px.bar(
        summary_df,
        x="Группа",
        y="Среднее",
        error_y="SE",
        text="Среднее",
        color="Группа",
        title=title
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# ==== Таблица под графиком ====
def display_summary_table(summary_df: pd.DataFrame):
    """Отображает таблицу со статистикой по группам."""
    st.dataframe(
        summary_df.style.format({
            "Среднее": "{:.2f}",
            "SD": "{:.2f}",
            "SE": "{:.2f}",
            "N": "{:d}"
        })
    )

# ==== Chi2 визуализация ====
def plot_chi2_table(table: pd.DataFrame, plot_choice: str = "Авто"):
    """Строит график для таблицы сопряжённости."""
    n_levels_x = table.shape[1]
    n_levels_y = table.shape[0]

    if plot_choice == "Авто":
        if n_levels_x <= 5 and n_levels_y <= 5:
            fig = px.imshow(
                table.values,
                labels=dict(
                    x=table.columns.name or "Col2",
                    y=table.index.name or "Col1",
                    color="Count"
                ),
                x=table.columns, y=table.index,
                text_auto=True, color_continuous_scale="Blues"
            )
        else:
            table_long = table.reset_index().melt(id_vars=table.index.name, value_name="count")
            fig = px.bar(
                table_long,
                x=table.columns.name, y="count", color=table.index.name,
                barmode="group", title="Сравнение категориальных частот"
            )
    elif plot_choice == "Heatmap":
        fig = px.imshow(
            table.values,
            labels=dict(
                x=table.columns.name or "Col2",
                y=table.index.name or "Col1",
                color="Count"
            ),
            x=table.columns, y=table.index,
            text_auto=True, color_continuous_scale="Blues"
        )
    elif plot_choice == "Stacked bar":
        table_long = table.reset_index().melt(id_vars=table.index.name, value_name="count")
        fig = px.bar(
            table_long,
            x=table.columns.name, y="count", color=table.index.name,
            barmode="stack", title="Stacked bar chart"
        )
    else:  # Clustered bar
        table_long = table.reset_index().melt(id_vars=table.index.name, value_name="count")
        fig = px.bar(
            table_long,
            x=table.columns.name, y="count", color=table.index.name,
            barmode="group", title="Clustered bar chart"
        )

    st.plotly_chart(fig, use_container_width=True)

# ==== T-test ====
# -----------------------------
# Вспомогательные функции
# -----------------------------

def _ci95(series):
    series = pd.Series(series).dropna()
    n = len(series)
    m = np.mean(series)
    s = np.std(series, ddof=1)
    if n < 2 or s == 0:
        return m, (m, m)
    se = s / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n-1)
    ci = (m - tcrit * se, m + tcrit * se)
    return m, ci

def _group_stats(df, target_col, group_col):
    levels = [lvl for lvl in df[group_col].dropna().unique().tolist()]
    stats_rows = []
    for lvl in levels:
        vals = df.loc[df[group_col] == lvl, target_col].dropna().values
        mean, ci = _ci95(vals)
        stats_rows.append({"group": lvl, "mean": mean, "ci_low": ci[0], "ci_high": ci[1], "n": len(vals)})
    return pd.DataFrame(stats_rows)

def _ttest(df, target_col, group_col, paired=False):
    levels = df[group_col].dropna().unique().tolist()
    if len(levels) != 2:
        return None, None, "❌ Нужно выбрать ровно две группы."
    g1 = df[df[group_col] == levels[0]][target_col].dropna().values
    g2 = df[df[group_col] == levels[1]][target_col].dropna().values
    if paired:
        if len(g1) != len(g2):
            return None, None, "❌ Для парного t‑теста размеры выборок должны совпадать."
        stat, pval = stats.ttest_rel(g1, g2)
    else:
        stat, pval = stats.ttest_ind(g1, g2, equal_var=False)
    return stat, pval, None

def _plot_box(df, target_col, group_col):
    return px.box(
        df, x=group_col, y=target_col, color=group_col,
        points="all",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Boxplot с точками"
    )

def _plot_bar_ci(df, target_col, group_col):
    statdf = _group_stats(df, target_col, group_col)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=statdf["group"],
        y=statdf["mean"],
        marker_color=px.colors.qualitative.Set2,
        name="Среднее"
    ))
    fig.add_trace(go.Scatter(
        x=statdf["group"], y=statdf["ci_low"], mode="markers",
        marker=dict(color="rgba(0,0,0,0.0)"), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=statdf["group"], y=statdf["ci_high"], mode="markers",
        marker=dict(color="rgba(0,0,0,0.0)"), showlegend=False
    ))
    # Ошибки как отрезки
    for i, row in statdf.iterrows():
        fig.add_shape(type="line",
                      x0=row["group"], x1=row["group"],
                      y0=row["ci_low"], y1=row["ci_high"],
                      line=dict(color="#333", width=2))
    fig.update_layout(title="Столбики со 95% ДИ",
                      yaxis_title=target_col, xaxis_title=group_col)
    return fig

def _plot_swarm(df, target_col, group_col):
    # Swarm/jitter: используем scatter с джиттером по x
    levels = df[group_col].dropna().unique().tolist()
    x_map = {lvl: i for i, lvl in enumerate(levels)}
    jitter = 0.15
    plot_df = df[[group_col, target_col]].dropna().copy()
    plot_df["_x"] = plot_df[group_col].map(x_map) + np.random.uniform(-jitter, jitter, size=len(plot_df))
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=[x_map[v] for v in plot_df[group_col]],
        y=plot_df[target_col],
        boxpoints=False, marker_color="rgba(0,0,0,0.0)",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["_x"], y=plot_df[target_col],
        mode="markers",
        marker=dict(size=7, color=plot_df[group_col].map(lambda v: px.colors.qualitative.Set2[x_map[v] % len(px.colors.qualitative.Set2)]), opacity=0.85),
        showlegend=False
    ))
    fig.update_layout(
        title="Swarm/Jitter с коробками",
        xaxis=dict(
            tickmode="array",
            tickvals=list(x_map.values()),
            ticktext=list(x_map.keys())
        ),
        yaxis_title=target_col,
        xaxis_title=group_col
    )
    return fig

def _plot_3d_scatter(df, target_col, group_col):
    # 3D scatter: ось X — группа, Y — значение, Z — индекс точки (для глубины)
    plot_df = df[[group_col, target_col]].dropna().copy().reset_index(drop=True)
    levels = plot_df[group_col].unique().tolist()
    x_map = {lvl: i for i, lvl in enumerate(levels)}
    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df[group_col].map(x_map),
        y=plot_df[target_col],
        z=np.arange(len(plot_df)),
        mode="markers",
        marker=dict(
            size=5,
            color=plot_df[group_col].map(lambda v: px.colors.qualitative.Set2[x_map[v] % len(px.colors.qualitative.Set2)]),
            opacity=0.9
        )
    )])
    fig.update_layout(
        title="3D Scatter: группы vs значения",
        scene=dict(
            xaxis_title=group_col,
            yaxis_title=target_col,
            zaxis_title="Индекс наблюдения",
            xaxis=dict(
                tickmode="array",
                tickvals=list(x_map.values()),
                ticktext=list(x_map.keys())
            )
        )
    )
    return fig

def _plot_3d_surface_density(df, target_col, group_col):
    # 3D surface: плотности для каждой группы по оси значений
    clean = df[[group_col, target_col]].dropna().copy()
    levels = clean[group_col].unique().tolist()
    if len(levels) < 2:
        # Если групп меньше двух, fallback на 3D scatter
        return _plot_3d_scatter(df, target_col, group_col)

    # общая сетка значений
    vals = clean[target_col].values
    xgrid = np.linspace(np.min(vals), np.max(vals), 80)

    z_rows = []
    y_positions = []
    colorscale = "Viridis"

    for iy, lvl in enumerate(levels):
        v = clean.loc[clean[group_col] == lvl, target_col].values
        if len(v) >= 2 and np.std(v) > 0:
            kde = gaussian_kde(v)
            dens = kde(xgrid)
        else:
            # если недостаточно точек для KDE — равномерная слабая плотность
            dens = np.ones_like(xgrid) * (1.0 / len(xgrid))
        z_rows.append(dens)
        y_positions.append(iy)

    Z = np.vstack(z_rows)  # shape: (n_groups, len(xgrid))
    X, Y = np.meshgrid(xgrid, y_positions)

    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z, colorscale=colorscale,
        contours={"z": {"show": True, "start": Z.min(), "end": Z.max(), "size": (Z.max()-Z.min())/6}}
    )])
    fig.update_layout(
        title="3D Surface: плотности значений по группам",
        scene=dict(
            xaxis_title=target_col,
            yaxis_title=group_col,
            zaxis_title="Плотность"
        )
    )
    # подписываем метки групп на оси Y
    fig.update_scenes(yaxis=dict(
        tickmode="array",
        tickvals=y_positions,
        ticktext=levels
    ))
    return fig

# -----------------------------
# Основные функции
# -----------------------------

def run_ttest_and_plot(df, target_col, group_col, paired, viz_type):
    # --- t-test ---
    stat, pval, err = _ttest(df, target_col, group_col, paired)
    if err:
        st.error(err)
        return
    st.markdown(f"**t‑статистика:** {stat:.3f} &nbsp;&nbsp; **p‑value:** {pval:.6f}")

    # --- Визуализация ---
    if viz_type == "Boxplot":
        fig = _plot_box(df, target_col, group_col)
    elif viz_type == "Bar + 95% CI":
        fig = _plot_bar_ci(df, target_col, group_col)
    elif viz_type == "Swarm/Jitter":
        fig = _plot_swarm(df, target_col, group_col)
    elif viz_type == "Histogram":
        fig = px.histogram(df, x=target_col, color=group_col,
                           barmode="overlay", opacity=0.6,
                           color_discrete_sequence=px.colors.qualitative.Set2,
                           title="Histogram: распределение значений по группам")
    elif viz_type == "3D Surface (Density)":
        fig = _plot_3d_surface_density(df, target_col, group_col)
    else:
        fig = _plot_box(df, target_col, group_col)

    st.plotly_chart(fig, use_container_width=True)

    # --- Таблица со статистикой ---
    desc = df.groupby(group_col)[target_col].describe().reset_index()
    st.dataframe(desc.style.format(precision=3))

def show_ttest_ui(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols_all = df.select_dtypes(exclude=["number"]).columns.tolist()

    if not num_cols:
        st.info("ℹ️ Нет числовых признаков для t‑test.")
        return
    if not cat_cols_all:
        st.info("ℹ️ Нет категориальных признаков для t‑test.")
        return

    st.subheader("⚙️ Настройки t‑test")
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Числовой признак (метрика)", num_cols, key="ttest_num")
    with col2:
        group_col = st.selectbox("Категориальный признак", cat_cols_all, key="ttest_group")

    levels = df[group_col].dropna().unique().tolist()

    if len(levels) >= 2:
        st.caption(f"Доступные группы: {', '.join(map(repr, levels))}")
    else:
        st.error("❌ Нужно минимум две группы для t‑теста.")
        return

    # выбор конкретных двух уровней (если больше двух)
    if len(levels) > 2:
        picked_levels = st.multiselect(
            "Выберите ДВА уровня для сравнения",
            options=levels, max_selections=2, key="ttest_levels"
        )
    else:
        picked_levels = levels

    paired = st.checkbox("Парный t‑test (paired)", value=False, key="ttest_paired")

    # выбор типа визуализации через selectbox
    viz_type = st.selectbox(
        "Тип визуализации",
        ["Boxplot", "Bar + 95% CI", "Swarm/Jitter", "Histogram", "3D Surface (Density)"],
        index=0,
        key="ttest_viz"
    )


    st.markdown("---")
    if st.button("▶️ Запустить t‑test", type="primary"):
        if len(picked_levels) == 2:
            df_pair = df[df[group_col].isin(picked_levels)].copy()
            run_ttest_and_plot(df_pair, target_col, group_col, paired, viz_type)
        else:
            st.error("❌ Выберите ровно две категории.")

# --- ANOVA ---
def run_anova(df, target_col, group_col):
    levels = df[group_col].dropna().unique().tolist()
    if len(levels) < 2:
        return None, None, "❌ Нужно минимум две группы для ANOVA."
    groups = [df[df[group_col] == lvl][target_col].dropna().values for lvl in levels]
    stat, pval = stats.f_oneway(*groups)
    return stat, pval, None

# --- Визуализации (используем те же вспомогательные функции, что и для t-test) ---
def run_anova_and_plot(df, target_col, group_col, viz_type):
    stat, pval, err = run_anova(df, target_col, group_col)
    if err:
        st.error(err)
        return
    st.markdown(f"**F‑статистика:** {stat:.3f} &nbsp;&nbsp; **p‑value:** {pval:.6f}")

    # --- Визуализация ---
    if viz_type == "Boxplot":
        fig = px.box(df, x=group_col, y=target_col, color=group_col,
                     points="all", color_discrete_sequence=px.colors.qualitative.Set2,
                     title="Boxplot по группам")
    elif viz_type == "Bar + 95% CI":
        statdf = df.groupby(group_col)[target_col].agg(["mean","count","std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=statdf[group_col], y=statdf["mean"],
            marker_color=px.colors.qualitative.Set2, name="Среднее"
        ))
        # Добавляем доверительные интервалы
        for i, row in statdf.iterrows():
            ci_low = row["mean"] - 1.96*row["std"]/np.sqrt(row["count"])
            ci_high = row["mean"] + 1.96*row["std"]/np.sqrt(row["count"])
            fig.add_shape(type="line", x0=row[group_col], x1=row[group_col],
                          y0=ci_low, y1=ci_high, line=dict(color="#333", width=2))
        fig.update_layout(title="Bar + 95% CI", yaxis_title=target_col)
    elif viz_type == "Swarm/Jitter":
        fig = px.strip(df, x=group_col, y=target_col, color=group_col,
                       color_discrete_sequence=px.colors.qualitative.Set2,
                       title="Swarm/Jitter по группам", stripmode="overlay")
    elif viz_type == "Histogram":
        fig = px.histogram(df, x=target_col, color=group_col,
                           barmode="overlay", opacity=0.6,
                           color_discrete_sequence=px.colors.qualitative.Set2,
                           title="Histogram: распределение значений по группам")
    elif viz_type == "3D Surface (Density)":
        # KDE плотности по группам
        clean = df[[group_col, target_col]].dropna().copy()
        levels = clean[group_col].unique().tolist()
        vals = clean[target_col].values
        xgrid = np.linspace(np.min(vals), np.max(vals), 80)
        z_rows, y_positions = [], []
        for iy, lvl in enumerate(levels):
            v = clean.loc[clean[group_col] == lvl, target_col].values
            if len(v) >= 2 and np.std(v) > 0:
                kde = gaussian_kde(v)
                dens = kde(xgrid)
            else:
                dens = np.ones_like(xgrid) * (1.0 / len(xgrid))
            z_rows.append(dens)
            y_positions.append(iy)
        Z = np.vstack(z_rows)
        X, Y = np.meshgrid(xgrid, y_positions)
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
        fig.update_layout(title="3D Surface: плотности по группам",
                          scene=dict(xaxis_title=target_col,
                                     yaxis_title=group_col,
                                     zaxis_title="Плотность"))
        fig.update_scenes(yaxis=dict(tickmode="array", tickvals=y_positions, ticktext=levels))
    else:
        fig = px.box(df, x=group_col, y=target_col, color=group_col)

    st.plotly_chart(fig, use_container_width=True)

    # --- Таблица со статистикой ---
    desc = df.groupby(group_col)[target_col].describe().reset_index()
    st.dataframe(desc.style.format(precision=3))

# --- UI ---
def show_anova_ui(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols_all = df.select_dtypes(exclude=["number"]).columns.tolist()

    if not num_cols:
        st.info("ℹ️ Нет числовых признаков для ANOVA.")
        return
    if not cat_cols_all:
        st.info("ℹ️ Нет категориальных признаков для ANOVA.")
        return

    st.subheader("⚙️ Настройки ANOVA")
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox("Числовой признак (метрика)", num_cols, key="anova_num")
    with col2:
        group_col = st.selectbox("Категориальный признак", cat_cols_all, key="anova_group")

    viz_type = st.selectbox(
        "Тип визуализации",
        ["Boxplot", "Bar + 95% CI", "Swarm/Jitter", "Histogram", "3D Surface (Density)"],
        index=0,
        key="anova_viz"
    )

    st.markdown("---")
    if st.button("▶️ Запустить ANOVA", type="primary"):
        run_anova_and_plot(df, target_col, group_col, viz_type)


# ==== Chi-squared ====
def run_chi2(df: pd.DataFrame, col1: str, col2: str, plot_choice: str = "Авто"):
    """Выполняет критерий хи‑квадрат Пирсона."""
    if not (is_categorical(df[col1]) and is_categorical(df[col2])):
        st.error("❌ Для Chi‑square нужны два категориальных признака.")
        return

    table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(table)

    display_test_result("Chi‑square", "Chi²‑статистика", chi2, p)
    plot_chi2_table(table, plot_choice)

    # Таблица наблюдаемых и ожидаемых значений
    expected_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    st.subheader("📊 Наблюдаемые значения")
    st.dataframe(table)
    st.subheader("📊 Ожидаемые значения")
    st.dataframe(expected_df.round(2))


def show_chi2_ui(df):
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if len(cat_cols) < 2:
        st.info("ℹ️ Для Chi‑square нужно минимум два категориальных признака.")
        return

    st.subheader("⚙️ Настройки Chi‑square")
    col1, col2 = st.columns(2)
    with col1:
        col1_val = st.selectbox("Категориальный признак №1", cat_cols, key="chi_col1")
    with col2:
        other_cats = [c for c in cat_cols if c != col1_val] or cat_cols
        col2_val = st.selectbox("Категориальный признак №2", other_cats, key="chi_col2")

    plot_choice = st.radio(
        "Тип графика",
        ["Авто", "Heatmap", "Stacked bar", "Clustered bar"],
        horizontal=True, key="chi_plot"
    )

    st.markdown("---")
    if st.button("▶️ Запустить Chi‑square", type="primary"):
        if col1_val == col2_val:
            st.error("❌ Выберите разные категориальные признаки.")
        else:
            run_chi2(df, col1_val, col2_val, plot_choice)