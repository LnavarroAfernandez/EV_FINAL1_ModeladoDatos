from __future__ import annotations

import os
from typing import Any

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update

from src.figures import (
    AMBER,
    BLUE,
    CYAN,
    GREEN,
    MUTED,
    PURPLE,
    RED,
    classification_confusion_figure,
    feature_importance_figure,
    funnel_figure,
    pca_figure,
    probability_gauge,
    reasons_figure,
    regression_scatter_figure,
    segment_share_figure,
    top_clients_figure,
    annual_trend_figure,
)
from src.services import (
    case_options,
    case_summary,
    filter_dashboard_data,
    format_currency,
    format_number,
    load_classification_importance,
    load_dashboard_data,
    load_metadata,
    load_model_use_matrix,
    load_recommendations,
    load_regression_importance,
    load_regression_predictions,
    load_segment_summary,
    load_segmented_clients,
    load_top_clients,
    predict_scenario,
    regression_subset_metrics,
    summary_metrics,
)

metadata = load_metadata()
dashboard_df = load_dashboard_data()
segment_summary_df = load_segment_summary()
segmented_clients_df = load_segmented_clients()
top_clients_df = load_top_clients()
classification_importance_df = load_classification_importance()
regression_importance_df = load_regression_importance()
regression_predictions_df = load_regression_predictions()
recommendations_df = load_recommendations()
model_use_df = load_model_use_matrix()

MIN_YEAR = int(dashboard_df["ANIO"].dropna().min())
MAX_YEAR = int(dashboard_df["ANIO"].dropna().max())
YEAR_MARKS = {
    year: str(year)
    for year in range(MIN_YEAR, MAX_YEAR + 1)
    if year in {MIN_YEAR, MAX_YEAR} or (year - MIN_YEAR) % 2 == 0
}
AREAS = sorted(dashboard_df["DES_AO"].dropna().astype(str).unique().tolist())
AREA_OPTIONS = [{"label": "Todas las áreas", "value": "Todas"}] + [
    {"label": area, "value": area} for area in AREAS
]
CASE_OPTIONS = case_options()
DEFAULT_CASE = CASE_OPTIONS[0]["value"] if CASE_OPTIONS else None

CLASSIFICATION = metadata["classification"]
BUSINESS = metadata["business_summary"]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Eagle Intelligence · Dashboard ML",
    suppress_callback_exceptions=True,
)
server = app.server


def section_title(title: str, subtitle: str | None = None) -> html.Div:
    children: list[Any] = [html.H2(title, className="section-title")]
    if subtitle:
        children.append(html.P(subtitle, className="section-subtitle"))
    return html.Div(children, className="section-heading")


def kpi_card(title: str, value_id: str, subtitle: str, accent: str = BLUE) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="kpi-title"),
                html.Div("—", id=value_id, className="kpi-value"),
                html.Div(subtitle, className="kpi-subtitle"),
            ]
        ),
        className="kpi-card h-100",
        style={"--accent": accent},
    )


def static_metric_card(title: str, value: str, subtitle: str, accent: str = BLUE) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="kpi-title"),
                html.Div(value, className="kpi-value"),
                html.Div(subtitle, className="kpi-subtitle"),
            ]
        ),
        className="kpi-card h-100",
        style={"--accent": accent},
    )


def profile_card(row: pd.Series, accent: str, strategy: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(row["Segmento"], className="profile-title", style={"color": accent}),
                html.P(
                    "Perfil obtenido mediante K-Means y traducido a una estrategia comercial.",
                    className="profile-description",
                ),
                html.Div(
                    [
                        html.Div([html.Span("Clientes"), html.Strong(f"{int(row['Clientes']):,}")]),
                        html.Div(
                            [
                                html.Span("Operaciones por cliente"),
                                html.Strong(f"{row['Operaciones por cliente (mediana)']:.0f}"),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("Servicios distintos"),
                                html.Strong(f"{row['Servicios distintos (mediana)']:.0f}"),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("Conversión HP → OI"),
                                html.Strong(f"{row['Conversión HP→OI (mediana %)']:.1f}%"),
                            ]
                        ),
                    ],
                    className="profile-stats",
                ),
                html.Div(strategy, className="profile-strategy", style={"borderColor": accent}),
            ]
        ),
        className="profile-card h-100",
        style={"--accent": accent},
    )


def recommendation_card(row: pd.Series) -> dbc.Card:
    colors = [RED, AMBER, GREEN, BLUE, PURPLE, CYAN]
    accent = colors[(int(row["Prioridad"]) - 1) % len(colors)]
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(f"{int(row['Prioridad']):02d}", className="recommendation-number", style={"color": accent}),
                html.H5(row["Recomendación"], className="recommendation-title"),
                html.P(row["Acción"], className="recommendation-text"),
            ]
        ),
        className="recommendation-card h-100",
        style={"--accent": accent},
    )


summary_tab = dbc.Container(
    [
        section_title(
            "Resumen comercial",
            "Del proceso HP → OI a indicadores que el equipo comercial puede interpretar y gestionar.",
        ),
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Periodo de análisis", className="control-label"),
                                dcc.RangeSlider(
                                    id="summary-year-range",
                                    min=MIN_YEAR,
                                    max=MAX_YEAR,
                                    step=1,
                                    value=[MIN_YEAR, MAX_YEAR],
                                    marks=YEAR_MARKS,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    allowCross=False,
                                ),
                            ],
                            lg=7,
                        ),
                        dbc.Col(
                            [
                                html.Label("Área o línea de servicio", className="control-label"),
                                dcc.Dropdown(
                                    id="summary-area",
                                    options=AREA_OPTIONS,
                                    value="Todas",
                                    clearable=False,
                                    className="dark-dropdown",
                                ),
                            ],
                            lg=3,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Aplicar filtros",
                                id="btn-summary",
                                n_clicks=0,
                                className="action-button w-100",
                            ),
                            lg=2,
                            className="d-flex align-items-end",
                        ),
                    ],
                    className="g-3",
                )
            ),
            className="filter-card mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(kpi_card("Registros analizados", "kpi-records", "Líneas de servicio", CYAN), xl=3, md=6),
                dbc.Col(kpi_card("Conversión a OI", "kpi-conversion", "Lectura por línea", GREEN), xl=3, md=6),
                dbc.Col(kpi_card("HP activas sin OI", "kpi-active", "Requieren seguimiento", AMBER), xl=3, md=6),
                dbc.Col(kpi_card("HP inactivas sin OI", "kpi-inactive", "Oportunidades cerradas", RED), xl=3, md=6),
            ],
            className="g-3 mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(kpi_card("Mediana de conversión", "kpi-days", "Días HP → OI válidos", PURPLE), xl=4, md=6),
                dbc.Col(kpi_card("Facturación histórica PEN", "kpi-pen", "No se mezcla con USD", BLUE), xl=4, md=6),
                dbc.Col(kpi_card("Facturación histórica USD", "kpi-usd", "No se mezcla con PEN", CYAN), xl=4, md=12),
            ],
            className="g-3 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="summary-funnel", config={"displayModeBar": False}), lg=6),
                dbc.Col(dcc.Graph(id="summary-reasons", config={"displayModeBar": False}), lg=6),
            ],
            className="g-4",
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id="summary-trend", config={"displayModeBar": False}), width=12),
            className="mt-2",
        ),
        dbc.Alert(
            [
                html.Strong("Lectura de negocio: "),
                "las propuestas que convierten suelen hacerlo rápido; las no convertidas concentran señales de precio, presupuesto y actualización del pipeline.",
            ],
            color="info",
            className="business-note",
        ),
    ],
    fluid=True,
    className="tab-container",
)

low_activity = segment_summary_df.loc[
    segment_summary_df["Segmento"].eq("Menor actividad y alta conversión")
].iloc[0]
high_value = segment_summary_df.loc[
    segment_summary_df["Segmento"].eq("Alto valor y alta actividad")
].iloc[0]

segmentation_tab = dbc.Container(
    [
        section_title(
            "Segmentación de clientes",
            "K-Means identifica dos perfiles; el dashboard traduce el resultado técnico a decisiones de cartera.",
        ),
        dbc.Row(
            [
                dbc.Col(
                    profile_card(
                        low_activity,
                        GREEN,
                        "Estrategia: venta cruzada, reactivación y aumento de frecuencia.",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    profile_card(
                        high_value,
                        AMBER,
                        "Estrategia: retención, gestión de cuentas clave y mejora del cierre.",
                    ),
                    lg=6,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        figure=segment_share_figure(segment_summary_df),
                        config={"displayModeBar": False},
                    ),
                    lg=7,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Hallazgo central", className="insight-label"),
                                html.Div("> 91%", className="insight-value"),
                                html.P(
                                    "de la facturación en ambas monedas se concentra en el segmento de alto valor y alta actividad.",
                                    className="insight-text",
                                ),
                                html.Hr(),
                                html.P(
                                    "La cartera no debe gestionarse con una única estrategia: el grupo de alto valor requiere protección y cierre; el grupo amplio requiere desarrollo.",
                                    className="insight-text",
                                ),
                            ]
                        ),
                        className="insight-card h-100",
                    ),
                    lg=5,
                ),
            ],
            className="g-4 mb-4",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Clientes principales por segmento", className="card-section-title"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Segmento", className="control-label"),
                                    dcc.Dropdown(
                                        id="segment-filter",
                                        options=[
                                            {"label": value, "value": value}
                                            for value in segment_summary_df["Segmento"]
                                        ],
                                        value="Alto valor y alta actividad",
                                        clearable=False,
                                    ),
                                ],
                                lg=6,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Moneda", className="control-label"),
                                    dcc.RadioItems(
                                        id="segment-currency",
                                        options=[
                                            {"label": "Soles", "value": "S/."},
                                            {"label": "Dólares", "value": "US$"},
                                        ],
                                        value="S/.",
                                        inline=True,
                                        className="radio-group",
                                    ),
                                ],
                                lg=6,
                            ),
                        ],
                        className="g-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(id="top-clients-chart", config={"displayModeBar": False}),
                                lg=7,
                            ),
                            dbc.Col(
                                dash_table.DataTable(
                                    id="top-clients-table",
                                    page_size=10,
                                    style_as_list_view=True,
                                    style_table={"overflowX": "auto"},
                                    style_header={
                                        "backgroundColor": "#132541",
                                        "color": "#E8F1FF",
                                        "fontWeight": "bold",
                                        "border": "none",
                                    },
                                    style_cell={
                                        "backgroundColor": "#0E1B32",
                                        "color": "#E8F1FF",
                                        "border": "none",
                                        "padding": "10px",
                                        "fontFamily": "Arial",
                                        "fontSize": 13,
                                        "textAlign": "left",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {"row_index": "odd"},
                                            "backgroundColor": "#101F38",
                                        }
                                    ],
                                ),
                                lg=5,
                                className="pt-4",
                            ),
                        ],
                        className="g-4 mt-1",
                    ),
                ]
            ),
            className="content-card mb-4",
        ),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.P(
                                            "La visualización PCA sirve únicamente para observar los grupos en dos dimensiones. El modelo fue entrenado con ocho indicadores estandarizados y con facturación PEN/USD separada.",
                                            className="technical-copy",
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(f"K seleccionado: {metadata['segmentation']['k']}"),
                                                html.Li(
                                                    f"Silhouette validado: {metadata['segmentation']['silhouette']:.4f}"
                                                ),
                                                html.Li(
                                                    f"Clientes segmentados: {metadata['segmentation']['clientes']:,}"
                                                ),
                                            ],
                                            className="technical-list",
                                        ),
                                    ],
                                    lg=4,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        figure=pca_figure(segmented_clients_df),
                                        config={"displayModeBar": False},
                                    ),
                                    lg=8,
                                ),
                            ],
                            className="g-4",
                        )
                    ],
                    title="Evidencia técnica de K-Means",
                )
            ],
            start_collapsed=True,
            className="technical-accordion",
        ),
    ],
    fluid=True,
    className="tab-container",
)

simulator_tab = dbc.Container(
    [
        section_title(
            "Simulador pre-OI",
            "Seleccione una HP histórica como punto de partida y evalúe un escenario modificando sus principales variables numéricas.",
        ),
        dbc.Alert(
            [
                html.Strong("Qué predice: "),
                "potencial de facturación alta relativa dentro de la moneda y monto estimado si la oportunidad llega a facturarse. No predice conversión ni garantiza una venta.",
            ],
            color="warning",
            className="business-note",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("1. Seleccione un caso", className="card-section-title"),
                                html.Label("HP / cliente / servicio", className="control-label"),
                                dcc.Dropdown(
                                    id="sim-case",
                                    options=CASE_OPTIONS,
                                    value=DEFAULT_CASE,
                                    clearable=False,
                                    searchable=True,
                                    placeholder="Buscar una HP histórica...",
                                ),
                                html.Div(id="sim-case-summary", className="case-summary mt-3"),
                                html.H4("2. Modifique el escenario", className="card-section-title mt-4"),
                                html.Label("Variación del importe bruto de la HP", className="control-label"),
                                dcc.Slider(
                                    id="sim-amount-change",
                                    min=-50,
                                    max=100,
                                    step=5,
                                    value=0,
                                    marks={-50: "-50%", 0: "0%", 50: "+50%", 100: "+100%"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                                html.Label("Variación de participantes", className="control-label mt-4"),
                                dcc.Slider(
                                    id="sim-participants-change",
                                    min=-50,
                                    max=200,
                                    step=10,
                                    value=0,
                                    marks={-50: "-50%", 0: "0%", 100: "+100%", 200: "+200%"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                                html.Label("Variación de duración", className="control-label mt-4"),
                                dcc.Slider(
                                    id="sim-duration-change",
                                    min=-50,
                                    max=100,
                                    step=5,
                                    value=0,
                                    marks={-50: "-50%", 0: "0%", 50: "+50%", 100: "+100%"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                                dbc.Button(
                                    "Evaluar escenario",
                                    id="btn-simulate",
                                    n_clicks=0,
                                    className="action-button w-100 mt-4",
                                ),
                            ]
                        ),
                        className="content-card h-100",
                    ),
                    lg=5,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Resultado del modelo", className="card-section-title"),
                                    dcc.Graph(
                                        id="sim-gauge",
                                        figure=probability_gauge(0.0, "Sin evaluar"),
                                        config={"displayModeBar": False},
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(static_metric_card("Clasificación", "—", "Umbral 50%", GREEN), md=6, id="sim-class-card"),
                                            dbc.Col(static_metric_card("Facturación estimada", "—", "Por moneda", BLUE), md=6, id="sim-estimate-card"),
                                        ],
                                        className="g-3 mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(static_metric_card("Rango de referencia", "—", "Percentiles de árboles", PURPLE), md=6, id="sim-range-card"),
                                            dbc.Col(static_metric_card("Umbral de facturación alta", "—", "Mediana del entrenamiento", AMBER), md=6, id="sim-threshold-card"),
                                        ],
                                        className="g-3",
                                    ),
                                    html.Div(id="sim-message", className="sim-message mt-3"),
                                ]
                            ),
                            className="content-card mb-4",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Detalle del escenario evaluado", className="card-section-title"),
                                    dash_table.DataTable(
                                        id="sim-details",
                                        style_as_list_view=True,
                                        style_header={
                                            "backgroundColor": "#132541",
                                            "color": "#E8F1FF",
                                            "fontWeight": "bold",
                                            "border": "none",
                                        },
                                        style_cell={
                                            "backgroundColor": "#0E1B32",
                                            "color": "#E8F1FF",
                                            "border": "none",
                                            "padding": "10px",
                                            "fontFamily": "Arial",
                                            "fontSize": 13,
                                            "textAlign": "left",
                                            "whiteSpace": "normal",
                                            "height": "auto",
                                        },
                                    ),
                                ]
                            ),
                            className="content-card",
                        ),
                    ],
                    lg=7,
                ),
            ],
            className="g-4",
        ),
    ],
    fluid=True,
    className="tab-container",
)

classification_confusion = classification_confusion_figure(CLASSIFICATION["Matriz_confusion"])
classification_importance = feature_importance_figure(
    classification_importance_df,
    "Factores más utilizados por el Random Forest pre-OI",
)

validation_tab = dbc.Container(
    [
        section_title(
            "Validación y desempeño",
            "La evaluación técnica se presenta en lenguaje de negocio y mantiene separados clasificación y regresión.",
        ),
        html.H3("Clasificación pre-OI", className="subsection-title"),
        dbc.Row(
            [
                dbc.Col(static_metric_card("Accuracy", f"{CLASSIFICATION['Accuracy']:.2%}", "Acierto global", BLUE), xl=2, md=4, sm=6),
                dbc.Col(static_metric_card("Precisión", f"{CLASSIFICATION['Precision']:.2%}", "Calidad de alertas altas", CYAN), xl=2, md=4, sm=6),
                dbc.Col(static_metric_card("Recall", f"{CLASSIFICATION['Recall']:.2%}", "Altas detectadas", AMBER), xl=2, md=4, sm=6),
                dbc.Col(static_metric_card("F1", f"{CLASSIFICATION['F1']:.2%}", "Equilibrio del modelo", GREEN), xl=2, md=4, sm=6),
                dbc.Col(static_metric_card("ROC-AUC", f"{CLASSIFICATION['ROC_AUC']:.2%}", "Capacidad de ranking", PURPLE), xl=2, md=4, sm=6),
                dbc.Col(static_metric_card("Registros", f"{CLASSIFICATION['Registros']:,}", "Base positiva facturada", RED), xl=2, md=4, sm=6),
            ],
            className="g-3 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=classification_confusion, config={"displayModeBar": False}), lg=5),
                dbc.Col(dcc.Graph(figure=classification_importance, config={"displayModeBar": False}), lg=7),
            ],
            className="g-4 mb-4",
        ),
        dbc.Alert(
            "La importancia muestra cuánto utiliza el modelo una variable para separar casos; no demuestra causalidad. El principal factor global es el importe bruto de la HP.",
            color="info",
            className="business-note",
        ),
        html.H3("Regresión por moneda", className="subsection-title mt-5"),
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Moneda", className="control-label"),
                                dcc.RadioItems(
                                    id="reg-currency",
                                    options=[
                                        {"label": "Soles", "value": "S/."},
                                        {"label": "Dólares", "value": "US$"},
                                    ],
                                    value="S/.",
                                    inline=True,
                                    className="radio-group",
                                ),
                            ],
                            lg=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Periodo del conjunto de prueba", className="control-label"),
                                dcc.RangeSlider(
                                    id="reg-year-range",
                                    min=MIN_YEAR,
                                    max=MAX_YEAR,
                                    step=1,
                                    value=[MIN_YEAR, MAX_YEAR],
                                    marks=YEAR_MARKS,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    allowCross=False,
                                ),
                            ],
                            lg=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Área", className="control-label"),
                                dcc.Dropdown(
                                    id="reg-area",
                                    options=AREA_OPTIONS,
                                    value="Todas",
                                    clearable=False,
                                ),
                            ],
                            lg=2,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Actualizar evaluación",
                                id="btn-reg-validation",
                                n_clicks=0,
                                className="action-button w-100",
                            ),
                            lg=2,
                            className="d-flex align-items-end",
                        ),
                    ],
                    className="g-3",
                )
            ),
            className="filter-card mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(kpi_card("Casos de prueba", "reg-kpi-records", "Después de filtros", CYAN), md=3),
                dbc.Col(kpi_card("MAE", "reg-kpi-mae", "Error absoluto medio", BLUE), md=3),
                dbc.Col(kpi_card("RMSE", "reg-kpi-rmse", "Penaliza errores grandes", AMBER), md=3),
                dbc.Col(kpi_card("R²", "reg-kpi-r2", "Capacidad explicativa", GREEN), md=3),
            ],
            className="g-3 mb-4",
        ),
        html.Div(id="reg-validation-caption", className="validation-caption mb-2"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="reg-scatter", config={"displayModeBar": True}), lg=7),
                dbc.Col(dcc.Graph(id="reg-importance", config={"displayModeBar": False}), lg=5),
            ],
            className="g-4",
        ),
        html.H3("Modelo → decisión", className="subsection-title mt-5"),
        dash_table.DataTable(
            data=model_use_df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in model_use_df.columns],
            style_as_list_view=True,
            style_header={
                "backgroundColor": "#132541",
                "color": "#E8F1FF",
                "fontWeight": "bold",
                "border": "none",
            },
            style_cell={
                "backgroundColor": "#0E1B32",
                "color": "#E8F1FF",
                "border": "none",
                "padding": "13px",
                "fontFamily": "Arial",
                "fontSize": 14,
                "textAlign": "left",
                "whiteSpace": "normal",
                "height": "auto",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#101F38"}
            ],
        ),
    ],
    fluid=True,
    className="tab-container",
)

conclusions_tab = dbc.Container(
    [
        section_title(
            "Conclusiones y recomendaciones",
            "Los hallazgos se convierten en acciones concretas de gestión comercial, financiera y de datos.",
        ),
        dbc.Row(
            [
                dbc.Col(static_metric_card("Conversión total HP", f"{BUSINESS['hp_conversion_total_pct']:.2f}%", "Lectura por HP analítica", GREEN), xl=3, md=6),
                dbc.Col(static_metric_card("Costo + precio", f"{BUSINESS['precio_costo_no_conversion_pct']:.1f}%", "Motivos de no conversión", AMBER), xl=3, md=6),
                dbc.Col(static_metric_card("Alto valor", "> 91%", "Facturación concentrada", CYAN), xl=3, md=6),
                dbc.Col(static_metric_card("RF pre-OI", f"{CLASSIFICATION['ROC_AUC']:.2%}", "ROC-AUC validado", BLUE), xl=3, md=6),
            ],
            className="g-3 mb-4",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Lectura final del proyecto", className="card-section-title"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5("El embudo ya puede medirse"),
                                    html.P("La base distingue propuestas convertidas, activas e inactivas y mide la velocidad de conversión."),
                                ],
                                lg=4,
                            ),
                            dbc.Col(
                                [
                                    html.H5("La cartera requiere estrategias distintas"),
                                    html.P("K-Means separa cuentas de alto valor de clientes con menor actividad y alta conversión."),
                                ],
                                lg=4,
                            ),
                            dbc.Col(
                                [
                                    html.H5("Random Forest es apoyo operativo"),
                                    html.P("Sirve para priorizar y estimar; no garantiza ventas ni reemplaza el criterio humano."),
                                ],
                                lg=4,
                            ),
                        ],
                        className="g-4 conclusion-copy",
                    ),
                ]
            ),
            className="content-card mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(recommendation_card(row), lg=4, md=6)
                for _, row in recommendations_df.iterrows()
            ],
            className="g-4 mb-4",
        ),
        dbc.Alert(
            [
                html.Strong("Recomendación de control: "),
                "no automatizar presupuestos con el modelo temporal mensual actual. Mantenerlo como análisis experimental hasta mejorar su desempeño y la calidad de la serie.",
            ],
            color="warning",
            className="business-note",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Límites que el usuario debe conocer", className="card-section-title"),
                    html.Ul([html.Li(item) for item in metadata["limitations"]], className="limitations-list"),
                ]
            ),
            className="content-card",
        ),
    ],
    fluid=True,
    className="tab-container",
)

app.layout = html.Div(
    [
        html.Header(
            dbc.Container(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.Div("E", className="brand-mark"),
                                    html.Div(
                                        [
                                            html.H1("EAGLE INTELLIGENCE", className="brand-title"),
                                            html.P(
                                                "Dashboard comercial y de Machine Learning · Eagle Consulting S.A.C.",
                                                className="brand-subtitle",
                                            ),
                                        ]
                                    ),
                                ],
                                className="brand-group",
                            ),
                            lg=9,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.Span("BASE OFICIAL", className="header-badge-label"),
                                    html.Strong("17,962 registros", className="header-badge-value"),
                                    html.Small("2012–2026", className="header-badge-small"),
                                ],
                                className="header-badge",
                            ),
                            lg=3,
                            className="d-flex justify-content-lg-end mt-3 mt-lg-0",
                        ),
                    ],
                    align="center",
                ),
                fluid=True,
                className="header-container",
            ),
            className="app-header",
        ),
        dbc.Tabs(
            [
                dbc.Tab(summary_tab, label="Resumen comercial", tab_id="tab-summary"),
                dbc.Tab(segmentation_tab, label="Segmentación", tab_id="tab-segmentation"),
                dbc.Tab(simulator_tab, label="Simulador pre-OI", tab_id="tab-simulator"),
                dbc.Tab(validation_tab, label="Validación", tab_id="tab-validation"),
                dbc.Tab(conclusions_tab, label="Conclusiones", tab_id="tab-conclusions"),
            ],
            id="main-tabs",
            active_tab="tab-summary",
            className="main-tabs",
        ),
        html.Footer(
            dbc.Container(
                [
                    html.Span("Fuente: Notebook V5 y 02_1_base_con_nuevos_datos.xlsx"),
                    html.Span("Los modelos apoyan decisiones; no reemplazan el criterio del negocio."),
                ],
                fluid=True,
                className="footer-content",
            ),
            className="app-footer",
        ),
    ],
    className="app-shell",
)


@app.callback(
    Output("kpi-records", "children"),
    Output("kpi-conversion", "children"),
    Output("kpi-active", "children"),
    Output("kpi-inactive", "children"),
    Output("kpi-days", "children"),
    Output("kpi-pen", "children"),
    Output("kpi-usd", "children"),
    Output("summary-funnel", "figure"),
    Output("summary-reasons", "figure"),
    Output("summary-trend", "figure"),
    Input("btn-summary", "n_clicks"),
    State("summary-year-range", "value"),
    State("summary-area", "value"),
)
def update_summary(_: int, year_range: list[int], area: str):
    filtered = filter_dashboard_data(year_range, area)
    metrics = summary_metrics(filtered)
    days = "—" if metrics["median_days"] is None else f"{metrics['median_days']:.0f} día(s)"
    return (
        f"{metrics['records']:,}",
        f"{metrics['conversion_pct']:.1f}%",
        f"{metrics['active']:,}",
        f"{metrics['inactive']:,}",
        days,
        format_currency(metrics["pen"], "S/.", 0),
        format_currency(metrics["usd"], "US$", 0),
        funnel_figure(filtered),
        reasons_figure(filtered),
        annual_trend_figure(filtered),
    )


@app.callback(
    Output("top-clients-table", "data"),
    Output("top-clients-table", "columns"),
    Output("top-clients-chart", "figure"),
    Input("segment-filter", "value"),
    Input("segment-currency", "value"),
)
def update_top_clients(segment: str, currency: str):
    subset = top_clients_df[
        top_clients_df["Segmento"].eq(segment) & top_clients_df["Moneda"].eq(currency)
    ].copy()
    subset = subset.sort_values("Posición").head(10)
    table = subset[["Posición", "Cliente", "Facturación"]].copy()
    table["Facturación"] = table["Facturación"].map(
        lambda value: format_currency(value, currency, 0)
    )
    columns = [{"name": col, "id": col} for col in table.columns]
    return table.to_dict("records"), columns, top_clients_figure(subset, currency)


@app.callback(
    Output("sim-case-summary", "children"),
    Output("sim-amount-change", "value"),
    Output("sim-participants-change", "value"),
    Output("sim-duration-change", "value"),
    Input("sim-case", "value"),
)
def load_simulator_case(case_id: str):
    summary = case_summary(case_id)
    children = [
        html.Div([html.Span("HP"), html.Strong(str(summary["hp"]))]),
        html.Div([html.Span("Fecha"), html.Strong(summary["fecha"])]),
        html.Div([html.Span("Cliente"), html.Strong(summary["cliente"])]),
        html.Div([html.Span("Servicio"), html.Strong(summary["servicio"])]),
        html.Div([html.Span("Área"), html.Strong(summary["area"])]),
        html.Div([html.Span("Importe base"), html.Strong(format_currency(summary["importe"], summary["moneda"], 0))]),
        html.Div([html.Span("Participantes"), html.Strong(format_number(summary["participantes"], 0))]),
        html.Div([html.Span("Duración"), html.Strong(f"{format_number(summary['duracion'], 0)}")]),
        html.Div([html.Span("Segmento del cliente"), html.Strong(summary["segmento"])]),
    ]
    return children, 0, 0, 0


def _result_metric_card(title: str, value: str, subtitle: str, accent: str) -> dbc.Card:
    return static_metric_card(title, value, subtitle, accent)


@app.callback(
    Output("sim-gauge", "figure"),
    Output("sim-class-card", "children"),
    Output("sim-estimate-card", "children"),
    Output("sim-range-card", "children"),
    Output("sim-threshold-card", "children"),
    Output("sim-message", "children"),
    Output("sim-details", "data"),
    Output("sim-details", "columns"),
    Input("btn-simulate", "n_clicks"),
    State("sim-case", "value"),
    State("sim-amount-change", "value"),
    State("sim-participants-change", "value"),
    State("sim-duration-change", "value"),
)
def run_simulator(
    n_clicks: int,
    case_id: str,
    amount_change: float,
    participants_change: float,
    duration_change: float,
):
    if case_id is None:
        return (no_update,) * 8
    result = predict_scenario(
        case_id,
        amount_change or 0,
        participants_change or 0,
        duration_change or 0,
    )
    class_text = "Facturación alta relativa" if result["classification"] else "Facturación baja/media relativa"
    gauge = probability_gauge(result["probability"], result["level"])
    class_card = _result_metric_card("Clasificación", class_text, "Umbral de probabilidad: 50%", GREEN if result["classification"] else AMBER)
    estimate_card = _result_metric_card(
        "Facturación estimada",
        format_currency(result["estimate"], result["currency"], 0),
        "Estimación en la moneda de la HP",
        BLUE,
    )
    range_card = _result_metric_card(
        "Rango de referencia",
        f"{format_currency(result['range_low'], result['currency'], 0)} – {format_currency(result['range_high'], result['currency'], 0)}",
        "P10–P90 de los árboles del bosque",
        PURPLE,
    )
    threshold_card = _result_metric_card(
        "Umbral de facturación alta",
        format_currency(result["threshold"], result["currency"], 0),
        "Mediana calculada solo con entrenamiento",
        AMBER,
    )
    message = [
        html.Strong(f"{result['level']}: "),
        result["message"],
        html.Br(),
        html.Small(
            "El resultado es un ranking relativo dentro de la moneda; no representa probabilidad de conversión.",
            className="muted-copy",
        ),
    ]
    details = pd.DataFrame(
        [
            {"Variable": "Cliente", "Valor": result["client"]},
            {"Variable": "Servicio", "Valor": result["service"]},
            {"Variable": "Área", "Valor": result["area"]},
            {"Variable": "Importe simulado", "Valor": format_currency(result["amount"], result["currency"], 0)},
            {"Variable": "Participantes simulados", "Valor": format_number(result["participants"], 0)},
            {"Variable": "Duración simulada", "Valor": format_number(result["duration"], 0)},
            {"Variable": "Probabilidad de alta relativa", "Valor": f"{result['probability']:.1%}"},
            {
                "Variable": "Facturación histórica del caso",
                "Valor": format_currency(result["actual_billing"], result["currency"], 0)
                if result["actual_billing"] is not None
                else "Sin facturación registrada",
            },
        ]
    )
    columns = [{"name": col, "id": col} for col in details.columns]
    return (
        gauge,
        class_card,
        estimate_card,
        range_card,
        threshold_card,
        message,
        details.to_dict("records"),
        columns,
    )


@app.callback(
    Output("reg-kpi-records", "children"),
    Output("reg-kpi-mae", "children"),
    Output("reg-kpi-rmse", "children"),
    Output("reg-kpi-r2", "children"),
    Output("reg-scatter", "figure"),
    Output("reg-importance", "figure"),
    Output("reg-validation-caption", "children"),
    Input("btn-reg-validation", "n_clicks"),
    State("reg-currency", "value"),
    State("reg-year-range", "value"),
    State("reg-area", "value"),
)
def update_regression_validation(
    _: int,
    currency: str,
    year_range: list[int],
    area: str,
):
    subset, metrics = regression_subset_metrics(currency, year_range, area)
    importance = regression_importance_df[
        regression_importance_df["Moneda"].eq(currency)
    ]
    mae = "—" if metrics["mae"] is None else format_currency(metrics["mae"], currency, 0)
    rmse = "—" if metrics["rmse"] is None else format_currency(metrics["rmse"], currency, 0)
    r2 = "—" if metrics["r2"] is None else f"{metrics['r2']:.3f}"
    area_text = "todas las áreas" if area == "Todas" else area
    caption = (
        f"Evaluación sobre datos de prueba no usados en el entrenamiento · {currency} · "
        f"{year_range[0]}–{year_range[1]} · {area_text}. "
        "Los filtros no reentrenan el modelo; solo permiten revisar su comportamiento en el subconjunto seleccionado."
    )
    return (
        f"{metrics['records']:,}",
        mae,
        rmse,
        r2,
        regression_scatter_figure(subset, currency),
        feature_importance_figure(
            importance,
            f"Importancia global de variables · {currency}",
            top_n=8,
        ),
        caption,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    debug = os.getenv("DASH_DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
