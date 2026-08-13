from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"

CURRENCY_TO_MODEL = {
    "S/.": MODELS_DIR / "rf_regresion_pen.joblib",
    "US$": MODELS_DIR / "rf_regresion_usd.joblib",
}


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    return json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_dashboard_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "dashboard_data.csv.gz",
        parse_dates=["Fecha", "FechaOI"],
        low_memory=False,
    )
    for column in [
        "ANIO",
        "numHP",
        "idUnidNeg",
        "duracion",
        "nParticipante",
        "ImporteB",
        "impFacturado",
        "dias_conversion_hp_oi",
        "diferencia_facturacion",
        "ingreso_neto_estimado",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["propuesta_convertida"] = df["propuesta_convertida"].fillna(False).astype(bool)
    df["case_id"] = df["case_id"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_cases() -> pd.DataFrame:
    df = pd.read_csv(
        ARTIFACTS_DIR / "casos_simulador.csv.gz",
        parse_dates=["Fecha", "FechaOI"],
        low_memory=False,
    )
    for column in ["ANIO", "numHP", "idUnidNeg", "duracion", "nParticipante", "ImporteB", "impFacturado"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["case_id"] = df["case_id"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_segmented_clients() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "clientes_segmentados.csv.gz")


@lru_cache(maxsize=1)
def load_segment_summary() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "resumen_segmentos.csv")


@lru_cache(maxsize=1)
def load_top_clients() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "top_clientes_segmento.csv")


@lru_cache(maxsize=1)
def load_classification_importance() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "importancia_clasificacion.csv")


@lru_cache(maxsize=1)
def load_regression_importance() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "importancia_regresion.csv")


@lru_cache(maxsize=1)
def load_regression_metrics() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "metricas_regresion.csv")


@lru_cache(maxsize=1)
def load_regression_predictions() -> pd.DataFrame:
    df = pd.read_csv(
        ARTIFACTS_DIR / "predicciones_regresion_prueba.csv.gz",
        parse_dates=["Fecha"],
        low_memory=False,
    )
    for column in ["ANIO", "real", "prediccion", "rango_p10", "rango_p90", "ImporteB"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


@lru_cache(maxsize=1)
def load_recommendations() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "recomendaciones.csv")


@lru_cache(maxsize=1)
def load_model_use_matrix() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS_DIR / "matriz_uso_modelos.csv")


@lru_cache(maxsize=1)
def load_classifier():
    return joblib.load(MODELS_DIR / "rf_clasificacion_pre_oi.joblib")


@lru_cache(maxsize=2)
def load_regressor(currency: str):
    if currency not in CURRENCY_TO_MODEL:
        raise ValueError(f"Moneda no soportada: {currency}")
    return joblib.load(CURRENCY_TO_MODEL[currency])


def currency_symbol(currency: str) -> str:
    return "S/" if currency == "S/." else "US$"


def format_number(value: float | int | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "No especificado"
    return f"{float(value):,.{decimals}f}"


def format_currency(value: float | int | None, currency: str, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "No especificado"
    return f"{currency_symbol(currency)} {float(value):,.{decimals}f}"


def shorten(text: Any, max_length: int = 52) -> str:
    value = str(text or "No especificado")
    if len(value) <= max_length:
        return value
    return value[: max_length - 1].rstrip() + "…"


def case_options(limit: int = 1500) -> list[dict[str, str]]:
    cases = load_cases().head(limit)
    options: list[dict[str, str]] = []
    for row in cases.itertuples(index=False):
        date_text = row.Fecha.strftime("%d/%m/%Y") if pd.notna(row.Fecha) else "Sin fecha"
        hp_text = str(int(row.numHP)) if pd.notna(row.numHP) else "N/E"
        label = (
            f"{date_text} · HP {hp_text} · {shorten(row.NomCliente, 30)} · "
            f"{shorten(row.Desc_Servicio, 35)} · {row.MON}"
        )
        options.append({"label": label, "value": str(row.case_id)})
    return options


def get_case(case_id: str | None) -> pd.Series:
    cases = load_cases()
    if case_id is None:
        return cases.iloc[0]
    match = cases[cases["case_id"].eq(str(case_id))]
    if match.empty:
        return cases.iloc[0]
    return match.iloc[0]


def case_summary(case_id: str | None) -> dict[str, Any]:
    row = get_case(case_id)
    return {
        "case_id": str(row["case_id"]),
        "fecha": row["Fecha"].strftime("%d/%m/%Y") if pd.notna(row["Fecha"]) else "No especificado",
        "hp": int(row["numHP"]) if pd.notna(row["numHP"]) else "No especificado",
        "cliente": row["NomCliente"],
        "servicio": row["Desc_Servicio"],
        "area": row["DES_AO"],
        "moneda": row["MON"],
        "importe": float(row["ImporteB"]) if pd.notna(row["ImporteB"]) else 0.0,
        "participantes": float(row["nParticipante"]) if pd.notna(row["nParticipante"]) else 0.0,
        "duracion": float(row["duracion"]) if pd.notna(row["duracion"]) else 15.0,
        "facturacion_real": float(row["impFacturado"]) if pd.notna(row["impFacturado"]) else None,
        "estado": row["Desc_Estado"],
        "segmento": row["segmento_negocio"],
    }


def _adjust(base_value: float, percent_change: float) -> float:
    return max(0.0, base_value * (1.0 + percent_change / 100.0))


def build_scenario(
    case_id: str | None,
    amount_change: float,
    participants_change: float,
    duration_change: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    row = get_case(case_id).copy()
    amount = _adjust(float(row["ImporteB"] or 0), float(amount_change or 0))
    participants = _adjust(float(row["nParticipante"] or 0), float(participants_change or 0))
    duration = max(1.0, _adjust(float(row["duracion"] or 15), float(duration_change or 0)))
    currency = row["MON"] if row["MON"] in {"S/.", "US$"} else "S/."

    scenario = pd.DataFrame(
        [
            {
                "ImporteB": amount,
                "nParticipante": participants,
                "duracion": duration,
                "ANIO": int(row["ANIO"]) if pd.notna(row["ANIO"]) else int(pd.Timestamp.today().year),
                "idUnidNeg": str(row["idUnidNeg"]),
                "DES_AO": str(row["DES_AO"]),
                "Desc_Servicio": str(row["Desc_Servicio"]),
                "NomCliente": str(row["NomCliente"]),
                "LugarEjec": str(row["LugarEjec"]),
                "Consultor": str(row["Consultor"]),
                "MON": currency,
                "Rubro": str(row["Rubro"]),
            }
        ]
    )
    details = {
        "currency": currency,
        "amount": amount,
        "participants": participants,
        "duration": duration,
        "client": row["NomCliente"],
        "service": row["Desc_Servicio"],
        "area": row["DES_AO"],
        "hp": int(row["numHP"]) if pd.notna(row["numHP"]) else "No especificado",
        "actual_billing": float(row["impFacturado"]) if pd.notna(row["impFacturado"]) else None,
    }
    return scenario, details


def predict_scenario(
    case_id: str | None,
    amount_change: float,
    participants_change: float,
    duration_change: float,
) -> dict[str, Any]:
    scenario, details = build_scenario(
        case_id, amount_change, participants_change, duration_change
    )
    metadata = load_metadata()
    classifier = load_classifier()
    probability = float(classifier.predict_proba(scenario)[:, 1][0])
    classification = int(probability >= 0.50)

    currency = details["currency"]
    regressor = load_regressor(currency)
    regression_columns = metadata["regression_features"]["numeric"] + metadata["regression_features"]["categorical"]
    regression_input = scenario[regression_columns].copy()
    estimate = float(regressor.predict(regression_input)[0])
    transformed = regressor.named_steps["preprocessor"].transform(regression_input)
    forest = regressor.named_steps["model"]
    tree_predictions = np.array([tree.predict(transformed)[0] for tree in forest.estimators_])
    low, high = np.quantile(tree_predictions, [0.10, 0.90])
    threshold = float(metadata["classification"]["Umbrales"][currency])

    if probability >= 0.75:
        message = (
            "El escenario aparece en el grupo de mayor potencial relativo. Conviene priorizar "
            "el seguimiento, validando disponibilidad, precio y condiciones comerciales."
        )
        level = "Potencial alto"
    elif probability >= 0.50:
        message = (
            "El escenario supera el punto de decisión, pero requiere revisión comercial antes de "
            "asignar prioridad alta."
        )
        level = "Potencial medio-alto"
    elif probability >= 0.30:
        message = (
            "El potencial estimado es intermedio. Puede mejorar con una propuesta mejor calificada "
            "o mayor información sobre el cliente y el servicio."
        )
        level = "Potencial intermedio"
    else:
        message = (
            "El escenario se parece más a operaciones de facturación baja o media dentro de su moneda. "
            "No significa que deba descartarse; indica menor prioridad relativa según el historial."
        )
        level = "Potencial bajo"

    return {
        **details,
        "probability": probability,
        "classification": classification,
        "level": level,
        "estimate": max(0.0, estimate),
        "range_low": max(0.0, float(low)),
        "range_high": max(0.0, float(high)),
        "threshold": threshold,
        "message": message,
    }


def filter_dashboard_data(year_range: list[int] | tuple[int, int], area: str) -> pd.DataFrame:
    df = load_dashboard_data()
    start_year, end_year = int(year_range[0]), int(year_range[1])
    observed_years = df["ANIO"].dropna()
    min_year = int(observed_years.min())
    max_year = int(observed_years.max())
    mask = df["ANIO"].between(start_year, end_year, inclusive="both")
    # Cuando se selecciona todo el periodo, se conserva también el único registro
    # sin año identificado para que el total coincida con la base oficial.
    if start_year <= min_year and end_year >= max_year:
        mask |= df["ANIO"].isna()
    if area and area != "Todas":
        mask &= df["DES_AO"].eq(area)
    return df.loc[mask].copy()


def summary_metrics(filtered: pd.DataFrame) -> dict[str, Any]:
    valid_days = filtered.loc[
        filtered["propuesta_convertida"] & filtered["dias_conversion_hp_oi"].ge(0),
        "dias_conversion_hp_oi",
    ].dropna()
    return {
        "records": int(len(filtered)),
        "conversion_pct": float(filtered["propuesta_convertida"].mean() * 100) if len(filtered) else 0.0,
        "active": int(filtered["etapa_comercial"].eq("HP activa sin OI").sum()),
        "inactive": int(filtered["etapa_comercial"].eq("HP inactiva sin OI").sum()),
        "median_days": float(valid_days.median()) if len(valid_days) else None,
        "pen": float(filtered.loc[filtered["MON"].eq("S/."), "impFacturado"].sum()),
        "usd": float(filtered.loc[filtered["MON"].eq("US$"), "impFacturado"].sum()),
    }


def regression_subset_metrics(
    currency: str,
    year_range: list[int] | tuple[int, int],
    area: str,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    df = load_regression_predictions()
    start_year, end_year = int(year_range[0]), int(year_range[1])
    subset = df[
        df["MON"].eq(currency)
        & df["ANIO"].between(start_year, end_year, inclusive="both")
    ].copy()
    if area and area != "Todas":
        subset = subset[subset["DES_AO"].eq(area)].copy()
    subset = subset.dropna(subset=["real", "prediccion"])
    if len(subset) < 2:
        return subset, {"records": len(subset), "mae": None, "rmse": None, "r2": None}
    metrics = {
        "records": int(len(subset)),
        "mae": float(mean_absolute_error(subset["real"], subset["prediccion"])),
        "rmse": float(mean_squared_error(subset["real"], subset["prediccion"]) ** 0.5),
        "r2": float(r2_score(subset["real"], subset["prediccion"])) if len(subset) > 2 else None,
    }
    return subset, metrics
