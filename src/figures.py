from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG = "#060B1D"
PANEL = "#0E1B32"
GRID = "rgba(140, 180, 230, 0.16)"
TEXT = "#E8F1FF"
MUTED = "#9CB0CA"
BLUE = "#2F8CFF"
CYAN = "#23D5D5"
GREEN = "#32D07C"
AMBER = "#F3C760"
RED = "#F0646C"
PURPLE = "#9B7CF6"


def apply_dark_layout(fig: go.Figure, *, height: int = 420, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Arial, sans-serif"},
        height=height,
        margin=margin or {"l": 50, "r": 25, "t": 55, "b": 50},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": MUTED}},
        hoverlabel={"bgcolor": PANEL, "font_color": TEXT},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def empty_figure(message: str, *, height: int = 380) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": MUTED},
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_dark_layout(fig, height=height)


def funnel_figure(df: pd.DataFrame) -> go.Figure:
    order = [
        "Convertida a OI",
        "HP activa sin OI",
        "HP inactiva sin OI",
        "Otro / no especificado",
    ]
    colors = {
        "Convertida a OI": GREEN,
        "HP activa sin OI": AMBER,
        "HP inactiva sin OI": RED,
        "Otro / no especificado": PURPLE,
    }
    counts = (
        df["etapa_comercial"]
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("Etapa")
        .reset_index(name="Registros")
    )
    total = max(int(counts["Registros"].sum()), 1)
    counts["Porcentaje"] = counts["Registros"] / total * 100
    fig = go.Figure(
        go.Bar(
            x=counts["Registros"],
            y=counts["Etapa"],
            orientation="h",
            marker_color=[colors[value] for value in counts["Etapa"]],
            text=[f"{n:,.0f} · {p:.1f}%" for n, p in zip(counts["Registros"], counts["Porcentaje"])],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Registros: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
    fig.update_layout(title="Embudo HP → OI por línea de servicio", showlegend=False)
    return apply_dark_layout(fig, height=390, margin={"l": 145, "r": 80, "t": 60, "b": 35})


def reasons_figure(df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    subset = df[
        (~df["propuesta_convertida"]) & df["Desc_Estado"].eq("HP Inactivo")
    ]
    if subset.empty:
        return empty_figure("No hay HP inactivas sin OI para los filtros seleccionados.")
    counts = (
        subset["motivo_negocio"]
        .value_counts()
        .head(top_n)
        .sort_values()
        .rename_axis("Motivo")
        .reset_index(name="Registros")
    )
    counts["Porcentaje"] = counts["Registros"] / len(subset) * 100
    fig = go.Figure(
        go.Bar(
            x=counts["Registros"],
            y=counts["Motivo"],
            orientation="h",
            marker_color=[AMBER if value in {"Costo de referencia", "Precio / presupuesto"} else BLUE for value in counts["Motivo"]],
            text=[f"{p:.1f}%" for p in counts["Porcentaje"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Registros: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title="Principales motivos de no conversión", showlegend=False)
    return apply_dark_layout(fig, height=390, margin={"l": 190, "r": 55, "t": 60, "b": 35})


def annual_trend_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("No hay datos para mostrar la evolución anual.")
    annual = (
        df.groupby("ANIO", dropna=True)
        .agg(Registros=("case_id", "size"), Conversion=("propuesta_convertida", "mean"))
        .reset_index()
        .sort_values("ANIO")
    )
    annual["Conversion_pct"] = annual["Conversion"] * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=annual["ANIO"],
            y=annual["Registros"],
            name="Registros",
            marker_color=BLUE,
            opacity=0.75,
            hovertemplate="Año %{x:.0f}<br>Registros: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=annual["ANIO"],
            y=annual["Conversion_pct"],
            name="Conversión",
            mode="lines+markers",
            line={"color": GREEN, "width": 3},
            marker={"size": 7},
            hovertemplate="Año %{x:.0f}<br>Conversión: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Registros", secondary_y=False)
    fig.update_yaxes(title_text="Conversión (%)", range=[0, 100], secondary_y=True)
    fig.update_layout(title="Actividad y conversión histórica por año", barmode="group")
    return apply_dark_layout(fig, height=420)


def segment_share_figure(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return empty_figure("No se encontró el resumen de segmentos.")
    metrics = ["% clientes", "% operaciones", "% facturación S/.", "% facturación US$"]
    labels = ["Clientes", "Operaciones", "Facturación S/.", "Facturación US$"]
    fig = go.Figure()
    palette = {
        "Menor actividad y alta conversión": GREEN,
        "Alto valor y alta actividad": AMBER,
    }
    for _, row in summary.iterrows():
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[row[m] for m in metrics],
                name=row["Segmento"],
                marker_color=palette.get(row["Segmento"], BLUE),
                text=[f"{row[m]:.1f}%" for m in metrics],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        title="Peso de cada segmento en la cartera",
        yaxis_title="Participación (%)",
        yaxis_range=[0, 100],
    )
    return apply_dark_layout(fig, height=430)


def pca_figure(clients: pd.DataFrame) -> go.Figure:
    if clients.empty:
        return empty_figure("No se encontró la evidencia PCA de la segmentación.")
    color_map = {
        "Menor actividad y alta conversión": GREEN,
        "Alto valor y alta actividad": AMBER,
    }
    fig = px.scatter(
        clients,
        x="PCA_1",
        y="PCA_2",
        color="segmento_negocio",
        hover_name="NomCliente",
        hover_data={
            "cantidad_registros": ":,.0f",
            "cantidad_servicios_distintos": ":,.0f",
            "tasa_conversion_hp_oi": ":.1%",
            "PCA_1": False,
            "PCA_2": False,
        },
        color_discrete_map=color_map,
        labels={
            "segmento_negocio": "Segmento",
            "cantidad_registros": "Operaciones",
            "cantidad_servicios_distintos": "Servicios distintos",
            "tasa_conversion_hp_oi": "Conversión",
        },
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.72, "line": {"width": 0}})
    fig.update_layout(title="Evidencia técnica: visualización PCA de los dos grupos")
    return apply_dark_layout(fig, height=430)


def top_clients_figure(df: pd.DataFrame, currency: str) -> go.Figure:
    if df.empty:
        return empty_figure("No hay clientes para el segmento y la moneda seleccionados.", height=360)
    subset = df.sort_values("Facturación", ascending=True).tail(10)
    fig = go.Figure(
        go.Bar(
            x=subset["Facturación"],
            y=subset["Cliente"],
            orientation="h",
            marker_color=CYAN,
            text=[f"{value:,.0f}" for value in subset["Facturación"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Facturación: %{x:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(title=f"Principales clientes del segmento · {currency}", showlegend=False)
    return apply_dark_layout(fig, height=410, margin={"l": 210, "r": 85, "t": 60, "b": 35})


def probability_gauge(probability: float, level: str) -> go.Figure:
    value = max(0.0, min(100.0, probability * 100))
    if value >= 75:
        color = GREEN
    elif value >= 50:
        color = CYAN
    elif value >= 30:
        color = AMBER
    else:
        color = RED
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 44, "color": TEXT}},
            title={"text": f"Potencial relativo<br><span style='font-size:0.72em;color:{MUTED}'>{level}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": PANEL,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(240,100,108,0.18)"},
                    {"range": [30, 50], "color": "rgba(243,199,96,0.18)"},
                    {"range": [50, 75], "color": "rgba(35,213,213,0.18)"},
                    {"range": [75, 100], "color": "rgba(50,208,124,0.18)"},
                ],
                "threshold": {
                    "line": {"color": TEXT, "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    return apply_dark_layout(fig, height=360, margin={"l": 30, "r": 30, "t": 55, "b": 20})


def classification_confusion_figure(matrix: list[list[int]]) -> go.Figure:
    values = np.array(matrix, dtype=int)
    labels = ["Baja/media", "Alta"]
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            colorscale=[[0, "#14233E"], [0.5, BLUE], [1, GREEN]],
            text=values,
            texttemplate="%{text:,.0f}",
            textfont={"size": 20},
            showscale=False,
            hovertemplate="Real: %{y}<br>Predicción: %{x}<br>Casos: %{z:,.0f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Predicción", side="top")
    fig.update_yaxes(title="Real", autorange="reversed")
    fig.update_layout(title="Matriz de confusión · modelo pre-OI")
    return apply_dark_layout(fig, height=390, margin={"l": 70, "r": 30, "t": 90, "b": 40})


def feature_importance_figure(df: pd.DataFrame, title: str, top_n: int = 8) -> go.Figure:
    if df.empty:
        return empty_figure("No se encontraron importancias de variables.")
    subset = df.sort_values("Importancia_%", ascending=True).tail(top_n)
    fig = go.Figure(
        go.Bar(
            x=subset["Importancia_%"],
            y=subset["Nombre_para_negocio"],
            orientation="h",
            marker_color=BLUE,
            text=[f"{value:.1f}%" for value in subset["Importancia_%"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Importancia: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(title=title, showlegend=False, xaxis_title="Importancia agregada (%)")
    return apply_dark_layout(fig, height=410, margin={"l": 190, "r": 70, "t": 60, "b": 45})


def regression_scatter_figure(df: pd.DataFrame, currency: str) -> go.Figure:
    if df.empty:
        return empty_figure("No hay suficientes registros de prueba para los filtros seleccionados.")
    max_value = float(max(df["real"].max(), df["prediccion"].max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["real"],
            y=df["prediccion"],
            mode="markers",
            marker={"color": CYAN, "size": 7, "opacity": 0.65},
            text=df["NomCliente"],
            customdata=np.stack([df["DES_AO"], df["numHP"]], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>HP: %{customdata[1]}<br>Área: %{customdata[0]}"
                "<br>Real: %{x:,.2f}<br>Estimado: %{y:,.2f}<extra></extra>"
            ),
            name="Casos de prueba",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode="lines",
            line={"color": AMBER, "dash": "dash"},
            name="Estimación perfecta",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=f"Facturación real frente a estimada · {currency}",
        xaxis_title="Facturación real",
        yaxis_title="Facturación estimada",
    )
    return apply_dark_layout(fig, height=440)
