from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "02_1_base_con_nuevos_datos.xlsx"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_CURRENCIES = ["S/.", "US$"]

NUMERIC_CLASS_FEATURES = ["ImporteB", "nParticipante", "duracion", "ANIO"]
CATEGORICAL_CLASS_FEATURES = [
    "idUnidNeg",
    "DES_AO",
    "Desc_Servicio",
    "NomCliente",
    "LugarEjec",
    "Consultor",
    "MON",
    "Rubro",
]
NUMERIC_REG_FEATURES = ["ImporteB", "nParticipante", "duracion", "ANIO"]
CATEGORICAL_REG_FEATURES = [
    "idUnidNeg",
    "DES_AO",
    "Desc_Servicio",
    "NomCliente",
    "LugarEjec",
    "Consultor",
    "Rubro",
]

BUSINESS_NAMES = {
    "ImporteB": "Importe bruto de la HP",
    "nParticipante": "Cantidad de participantes",
    "duracion": "Duración registrada",
    "ANIO": "Año de generación de la HP",
    "idUnidNeg": "Unidad de negocio",
    "DES_AO": "Área o línea de servicio",
    "Desc_Servicio": "Servicio cotizado",
    "NomCliente": "Cliente",
    "LugarEjec": "Lugar de ejecución",
    "Consultor": "Consultor responsable",
    "MON": "Moneda",
    "Rubro": "Rubro",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.upper().strip().split())


def categorize_non_conversion_reason(value: object) -> str:
    text = normalize_text(value)
    if not text or text == "NO ESPECIFICADO":
        return "No especificado"
    if "COSTO DE REFERENCIA" in text or "COSTO REFERENCIAL" in text:
        return "Costo de referencia"
    if "PRECIO" in text or "PRESUPUESTO" in text or "COSTO ELEVADO" in text:
        return "Precio / presupuesto"
    if "STAND BY" in text or "STANDBY" in text:
        return "Proyecto en stand by"
    if "AUSENCIA DE RESPUESTA" in text or "VENCIMIENTO" in text or "SIN RESPUESTA" in text:
        return "Ausencia de respuesta / vencimiento"
    if "OTRO PROVEEDOR" in text:
        return "Eligió otro proveedor"
    if "NO CUENTA CON EQUIPO" in text or "NO CONTABA CON EQUIPO" in text:
        return "No cuenta con equipo"
    if (
        ("DISPONIBILIDAD" in text or "HABILITACION" in text)
        and ("INSPECTOR" in text or "INSTRUCTOR" in text)
    ):
        return "Disponibilidad / habilitación de personal"
    if "OTRA HP" in text or "OTRA OI" in text:
        return "Atendido con otra HP/OI"
    if "NO REQUIER" in text or "NO TOMAR" in text:
        return "Servicio no requerido"
    return "Otros motivos"


def make_onehot(min_frequency: int) -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=True,
        )
    except TypeError:  # compatibility with older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def original_feature_name(transformed: str, categorical: Iterable[str]) -> str:
    if transformed.startswith("num__"):
        return transformed.split("__", 1)[1]
    if transformed.startswith("cat__"):
        remainder = transformed.split("__", 1)[1]
        for variable in categorical:
            if remainder.startswith(variable + "_"):
                return variable
        return remainder
    return transformed


def aggregate_feature_importance(
    pipeline: Pipeline,
    categorical_features: list[str],
    currency: str | None = None,
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()
    values = model.feature_importances_
    frame = pd.DataFrame(
        {"Variable_transformada": transformed_names, "Importancia": values}
    )
    frame["Variable_tecnica"] = frame["Variable_transformada"].map(
        lambda name: original_feature_name(name, categorical_features)
    )
    frame = (
        frame.groupby("Variable_tecnica", as_index=False)["Importancia"]
        .sum()
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )
    frame["Nombre_para_negocio"] = frame["Variable_tecnica"].map(BUSINESS_NAMES)
    frame["Importancia_%"] = frame["Importancia"] * 100
    if currency is not None:
        frame.insert(0, "Moneda", currency)
    return frame


def tree_prediction_interval(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    transformed = pipeline.named_steps["preprocessor"].transform(X)
    forest = pipeline.named_steps["model"]
    all_predictions = np.vstack([tree.predict(transformed) for tree in forest.estimators_])
    return np.quantile(all_predictions, 0.10, axis=0), np.quantile(all_predictions, 0.90, axis=0)


def safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def load_and_prepare_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró la base oficial: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    for column in ["Fecha", "FechaOI", "Fecha_dt", "FechaOI_dt"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    for column in [
        "ImporteB",
        "impFacturado",
        "nParticipante",
        "duracion",
        "ANIO",
        "dias_conversion_hp_oi",
        "diferencia_facturacion",
        "ingreso_neto_estimado",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in [
        "idUnidNeg",
        "DES_AO",
        "Desc_Servicio",
        "NomCliente",
        "LugarEjec",
        "Consultor",
        "MON",
        "Rubro",
        "Desc_Estado",
        "DescAnulado",
    ]:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .fillna("No especificado")
                .str.strip()
                .replace({"": "No especificado"})
                .astype("object")
            )
    df["propuesta_convertida"] = df["propuesta_convertida"].fillna(False).astype(bool)
    df["etapa_comercial"] = np.select(
        [
            df["propuesta_convertida"],
            (~df["propuesta_convertida"]) & df["Desc_Estado"].eq("HP Activo"),
            (~df["propuesta_convertida"]) & df["Desc_Estado"].eq("HP Inactivo"),
        ],
        ["Convertida a OI", "HP activa sin OI", "HP inactiva sin OI"],
        default="Otro / no especificado",
    )
    df["motivo_negocio"] = df["DescAnulado"].map(categorize_non_conversion_reason)
    return df


def build_business_summary(df: pd.DataFrame) -> dict:
    base_hp = df[df["Fecha"].notna() & df["numHP"].notna()].copy()
    base_hp["hp_id_analitico"] = (
        base_hp["Fecha"].dt.strftime("%Y-%m-%d")
        + "|"
        + base_hp["numHP"].astype("Int64").astype(str)
        + "|"
        + base_hp["NomCliente"].astype(str)
    )
    hp_summary = (
        base_hp.groupby("hp_id_analitico", as_index=False)
        .agg(
            fecha_hp=("Fecha", "min"),
            numHP=("numHP", "first"),
            cliente=("NomCliente", "first"),
            lineas_servicio=("hp_id_analitico", "size"),
            servicios_distintos=("Desc_Servicio", "nunique"),
            lineas_convertidas=("propuesta_convertida", "sum"),
        )
    )
    hp_summary["estado_conversion_hp"] = np.select(
        [
            hp_summary["lineas_convertidas"].eq(0),
            hp_summary["lineas_convertidas"].eq(hp_summary["lineas_servicio"]),
        ],
        ["No convertida", "Conversión total"],
        default="Conversión parcial",
    )
    hp_summary.to_csv(ARTIFACTS_DIR / "hp_analiticas.csv.gz", index=False, compression="gzip")

    valid_days = df.loc[df["propuesta_convertida"], "dias_conversion_hp_oi"].dropna()
    negative_dates = int((valid_days < 0).sum())
    valid_days = valid_days[valid_days >= 0]

    inactive = df[
        (~df["propuesta_convertida"]) & df["Desc_Estado"].eq("HP Inactivo")
    ].copy()
    reasons = (
        inactive["motivo_negocio"]
        .value_counts()
        .rename_axis("Motivo")
        .reset_index(name="Registros")
    )
    reasons["Porcentaje"] = reasons["Registros"] / len(inactive) * 100
    reasons.to_csv(ARTIFACTS_DIR / "motivos_no_conversion.csv", index=False, encoding="utf-8-sig")

    price_cost_pct = float(
        reasons.loc[
            reasons["Motivo"].isin(["Costo de referencia", "Precio / presupuesto"]),
            "Porcentaje",
        ].sum()
    )

    cutoff_date = df["Fecha"].max()
    active = df[
        (~df["propuesta_convertida"]) & df["Desc_Estado"].eq("HP Activo")
    ].copy()
    active["antiguedad_dias"] = (cutoff_date - active["Fecha"]).dt.days
    active_over_90 = int((active["antiguedad_dias"] > 90).sum())
    active_over_90_pct = float(active_over_90 / len(active) * 100) if len(active) else 0.0

    funnel_order = [
        "Convertida a OI",
        "HP activa sin OI",
        "HP inactiva sin OI",
        "Otro / no especificado",
    ]
    funnel = (
        df["etapa_comercial"]
        .value_counts()
        .reindex(funnel_order, fill_value=0)
        .rename_axis("Etapa")
        .reset_index(name="Registros")
    )
    funnel["Porcentaje"] = funnel["Registros"] / len(df) * 100
    funnel.to_csv(ARTIFACTS_DIR / "embudo_lineas.csv", index=False, encoding="utf-8-sig")

    portfolio_rows = []
    for area, group in df.groupby("DES_AO", dropna=False):
        portfolio_rows.append(
            {
                "Área / línea de servicio": area,
                "Registros": len(group),
                "Conversión (%)": group["propuesta_convertida"].mean() * 100,
                "Facturación S/.": group.loc[group["MON"].eq("S/."), "impFacturado"].sum(),
                "Facturación US$": group.loc[group["MON"].eq("US$"), "impFacturado"].sum(),
            }
        )
    portfolio = pd.DataFrame(portfolio_rows).sort_values("Registros", ascending=False)
    portfolio.to_csv(ARTIFACTS_DIR / "portafolio_servicios.csv", index=False, encoding="utf-8-sig")

    return {
        "total_registros": int(len(df)),
        "fecha_min": df["Fecha"].min().strftime("%Y-%m-%d"),
        "fecha_max": df["Fecha"].max().strftime("%Y-%m-%d"),
        "lineas_convertidas": int(df["propuesta_convertida"].sum()),
        "conversion_lineas_pct": float(df["propuesta_convertida"].mean() * 100),
        "hp_analiticas": int(len(hp_summary)),
        "hp_conversion_total": int((hp_summary["estado_conversion_hp"] == "Conversión total").sum()),
        "hp_conversion_total_pct": float(
            (hp_summary["estado_conversion_hp"] == "Conversión total").mean() * 100
        ),
        "hp_conversion_parcial": int((hp_summary["estado_conversion_hp"] == "Conversión parcial").sum()),
        "hp_no_convertidas": int((hp_summary["estado_conversion_hp"] == "No convertida").sum()),
        "mediana_dias_conversion": safe_float(valid_days.median()),
        "promedio_dias_conversion": safe_float(valid_days.mean()),
        "conversion_15_dias_pct": safe_float((valid_days <= 15).mean() * 100),
        "conversion_30_dias_pct": safe_float((valid_days <= 30).mean() * 100),
        "p90_dias_conversion": safe_float(valid_days.quantile(0.90)),
        "fechas_inconsistentes": negative_dates,
        "precio_costo_no_conversion_pct": price_cost_pct,
        "hp_activas_sin_oi": int(len(active)),
        "hp_activas_mas_90": active_over_90,
        "hp_activas_mas_90_pct": active_over_90_pct,
        "total_facturacion_pen": safe_float(df.loc[df["MON"].eq("S/."), "impFacturado"].sum()),
        "total_facturacion_usd": safe_float(df.loc[df["MON"].eq("US$"), "impFacturado"].sum()),
    }


def train_classification(df: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame, pd.DataFrame]:
    base = df[
        df["impFacturado"].notna()
        & (df["impFacturado"] > 0)
        & df["MON"].isin(VALID_CURRENCIES)
    ].copy()
    for column in NUMERIC_CLASS_FEATURES:
        base[column] = pd.to_numeric(base[column], errors="coerce")
    for column in CATEGORICAL_CLASS_FEATURES:
        base[column] = (
            base[column].astype("string").fillna("No especificado").astype("object")
        )
    base = base[base["ImporteB"].notna()].copy()

    idx_train, idx_test = train_test_split(
        base.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=base["MON"],
    )
    train = base.loc[idx_train].copy()
    test = base.loc[idx_test].copy()
    thresholds = train.groupby("MON")["impFacturado"].median().to_dict()
    train["target"] = train.apply(
        lambda row: int(row["impFacturado"] > thresholds[row["MON"]]), axis=1
    )
    test["target"] = test.apply(
        lambda row: int(row["impFacturado"] > thresholds[row["MON"]]), axis=1
    )

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_CLASS_FEATURES,
            ),
            (
                "cat",
                Pipeline([("onehot", make_onehot(10))]),
                CATEGORICAL_CLASS_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    feature_columns = NUMERIC_CLASS_FEATURES + CATEGORICAL_CLASS_FEATURES
    X_train = train[feature_columns]
    X_test = test[feature_columns]
    y_train = train["target"]
    y_test = test["target"]
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Modelo": "Random Forest pre-OI",
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_test, proba)),
        "Registros": int(len(base)),
        "Muestra_prueba": int(len(test)),
        "Umbrales": {currency: float(value) for currency, value in thresholds.items()},
        "Matriz_confusion": confusion_matrix(y_test, pred).tolist(),
    }

    test_predictions = test[
        ["Fecha", "ANIO", "MON", "DES_AO", "NomCliente", "numHP", "impFacturado"]
    ].copy()
    test_predictions["real"] = y_test.values
    test_predictions["prediccion"] = pred
    test_predictions["probabilidad_alta"] = proba
    test_predictions.to_csv(
        ARTIFACTS_DIR / "predicciones_clasificacion_prueba.csv.gz",
        index=False,
        compression="gzip",
    )

    importances = aggregate_feature_importance(pipeline, CATEGORICAL_CLASS_FEATURES)
    return pipeline, metrics, importances, test_predictions


def train_regression(df: pd.DataFrame) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = df[
        df["impFacturado"].notna()
        & (df["impFacturado"] > 0)
        & df["MON"].isin(VALID_CURRENCIES)
    ].copy()
    for column in NUMERIC_REG_FEATURES:
        base[column] = pd.to_numeric(base[column], errors="coerce")
    for column in CATEGORICAL_REG_FEATURES:
        base[column] = (
            base[column].astype("string").fillna("No especificado").astype("object")
        )
    base = base[base["ImporteB"].notna()].copy()

    models: dict[str, Pipeline] = {}
    metric_rows: list[dict] = []
    importance_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for currency, group in base.groupby("MON"):
        X = group[NUMERIC_REG_FEATURES + CATEGORICAL_REG_FEATURES].copy()
        y = group["impFacturado"].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        min_frequency = 5 if currency == "US$" else 10
        preprocessor = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    NUMERIC_REG_FEATURES,
                ),
                (
                    "cat",
                    Pipeline([("onehot", make_onehot(min_frequency))]),
                    CATEGORICAL_REG_FEATURES,
                ),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        low, high = tree_prediction_interval(pipeline, X_test)
        metric_rows.append(
            {
                "Moneda": currency,
                "Modelo": "Random Forest pre-OI",
                "Registros": int(len(group)),
                "Muestra_prueba": int(len(X_test)),
                "MAE_prueba": float(mean_absolute_error(y_test, pred)),
                "RMSE_prueba": float(mean_squared_error(y_test, pred) ** 0.5),
                "R2_prueba": float(r2_score(y_test, pred)),
            }
        )
        importance_frames.append(
            aggregate_feature_importance(pipeline, CATEGORICAL_REG_FEATURES, currency)
        )
        frame = group.loc[X_test.index, [
            "Fecha", "ANIO", "MON", "DES_AO", "NomCliente", "numHP", "ImporteB", "impFacturado"
        ]].copy()
        frame["real"] = y_test.values
        frame["prediccion"] = pred
        frame["rango_p10"] = low
        frame["rango_p90"] = high
        prediction_frames.append(frame)
        models[currency] = pipeline

    metrics = pd.DataFrame(metric_rows)
    importances = pd.concat(importance_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        ARTIFACTS_DIR / "predicciones_regresion_prueba.csv.gz",
        index=False,
        compression="gzip",
    )
    return models, metrics, importances, predictions


def build_segments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model_df = df.copy()
    model_df["NomCliente"] = model_df["NomCliente"].astype(str).str.strip()
    model_df = model_df[
        model_df["NomCliente"].notna()
        & model_df["NomCliente"].ne("")
        & model_df["NomCliente"].str.lower().ne("nan")
        & model_df["NomCliente"].str.lower().ne("no especificado")
    ].copy()
    model_df["propuesta_convertida_num"] = model_df["propuesta_convertida"].astype(int)
    model_df["MON_limpia"] = model_df["MON"].where(model_df["MON"].isin(VALID_CURRENCIES), "Otra")
    for column in ["impFacturado", "ImporteB", "diferencia_facturacion", "nParticipante", "duracion"]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    invoice_by_currency = (
        model_df.pivot_table(
            index="NomCliente",
            columns="MON_limpia",
            values="impFacturado",
            aggfunc=lambda series: series.sum(min_count=1),
            fill_value=0,
        )
        .reset_index()
    )
    for currency, name in {
        "S/.": "facturacion_soles_total",
        "US$": "facturacion_dolares_total",
    }.items():
        if currency not in invoice_by_currency.columns:
            invoice_by_currency[currency] = 0.0
        invoice_by_currency[name] = pd.to_numeric(
            invoice_by_currency[currency], errors="coerce"
        ).fillna(0)

    model_df["porcentaje_diferencia_operacion"] = np.where(
        model_df["ImporteB"].notna() & model_df["ImporteB"].ne(0),
        model_df["diferencia_facturacion"] / model_df["ImporteB"] * 100,
        np.nan,
    )
    operational = (
        model_df.groupby("NomCliente")
        .agg(
            porcentaje_diferencia_mediana=("porcentaje_diferencia_operacion", "median"),
            participantes_total=("nParticipante", lambda s: s.sum(min_count=1)),
            duracion_promedio=("duracion", "mean"),
            tasa_conversion_hp_oi=("propuesta_convertida_num", "mean"),
            cantidad_servicios_distintos=("Desc_Servicio", "nunique"),
            cantidad_registros=("NomCliente", "size"),
        )
        .reset_index()
    )
    clients = operational.merge(
        invoice_by_currency[
            ["NomCliente", "facturacion_soles_total", "facturacion_dolares_total"]
        ],
        on="NomCliente",
        how="left",
    )
    variables = [
        "facturacion_soles_total",
        "facturacion_dolares_total",
        "porcentaje_diferencia_mediana",
        "participantes_total",
        "duracion_promedio",
        "tasa_conversion_hp_oi",
        "cantidad_servicios_distintos",
        "cantidad_registros",
    ]
    clients = clients[
        (clients["facturacion_soles_total"].fillna(0) + clients["facturacion_dolares_total"].fillna(0)) > 0
    ].copy()
    clients = clients[["NomCliente"] + variables].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = clients[variables].copy()
    log_variables = [
        "facturacion_soles_total",
        "facturacion_dolares_total",
        "participantes_total",
        "cantidad_servicios_distintos",
        "cantidad_registros",
    ]
    for column in log_variables:
        X[column] = np.log1p(X[column])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    max_k = min(10, len(clients) - 1)
    scores = []
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_scaled)
        scores.append({"k": k, "silhouette_score": silhouette_score(X_scaled, labels)})
    score_df = pd.DataFrame(scores)
    selected_k = int(score_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    kmeans = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=10)
    clients["cluster_validado"] = kmeans.fit_predict(X_scaled)
    median_activity = clients.groupby("cluster_validado")["cantidad_registros"].median()
    high_activity_cluster = int(median_activity.idxmax())
    clients["segmento_negocio"] = clients["cluster_validado"].map(
        lambda c: "Alto valor y alta actividad" if int(c) == high_activity_cluster else "Menor actividad y alta conversión"
    )
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    clients["PCA_1"] = coords[:, 0]
    clients["PCA_2"] = coords[:, 1]

    rows = model_df.merge(
        clients[["NomCliente", "cluster_validado", "segmento_negocio"]],
        on="NomCliente",
        how="inner",
    )
    total_clients = len(clients)
    total_rows = len(rows)
    total_pen = rows.loc[rows["MON_limpia"].eq("S/."), "impFacturado"].sum()
    total_usd = rows.loc[rows["MON_limpia"].eq("US$"), "impFacturado"].sum()
    summary_rows = []
    for cluster, group_clients in clients.groupby("cluster_validado"):
        segment = group_clients["segmento_negocio"].iloc[0]
        group_rows = rows[rows["cluster_validado"].eq(cluster)]
        pen = group_rows.loc[group_rows["MON_limpia"].eq("S/."), "impFacturado"].sum()
        usd = group_rows.loc[group_rows["MON_limpia"].eq("US$"), "impFacturado"].sum()
        summary_rows.append(
            {
                "Segmento": segment,
                "Clientes": int(len(group_clients)),
                "% clientes": len(group_clients) / total_clients * 100,
                "% operaciones": len(group_rows) / total_rows * 100,
                "% facturación S/.": pen / total_pen * 100 if total_pen else np.nan,
                "% facturación US$": usd / total_usd * 100 if total_usd else np.nan,
                "Operaciones por cliente (mediana)": float(group_clients["cantidad_registros"].median()),
                "Servicios distintos (mediana)": float(group_clients["cantidad_servicios_distintos"].median()),
                "Participantes (mediana)": float(group_clients["participantes_total"].median()),
                "Conversión HP→OI (mediana %)": float(group_clients["tasa_conversion_hp_oi"].median() * 100),
                "Duración (mediana)": float(group_clients["duracion_promedio"].median()),
                "Facturación S/. por cliente (mediana)": float(group_clients["facturacion_soles_total"].median()),
                "Facturación US$ por cliente (mediana)": float(group_clients["facturacion_dolares_total"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    top_rows = []
    for segment, group in rows.groupby("segmento_negocio"):
        for currency in VALID_CURRENCIES:
            top = (
                group[group["MON_limpia"].eq(currency)]
                .groupby("NomCliente")["impFacturado"]
                .sum()
                .sort_values(ascending=False)
                .head(25)
            )
            for rank, (client, amount) in enumerate(top.items(), start=1):
                top_rows.append(
                    {
                        "Segmento": segment,
                        "Moneda": currency,
                        "Posición": rank,
                        "Cliente": client,
                        "Facturación": float(amount),
                    }
                )
    pd.DataFrame(top_rows).to_csv(
        ARTIFACTS_DIR / "top_clientes_segmento.csv", index=False, encoding="utf-8-sig"
    )

    clients.to_csv(
        ARTIFACTS_DIR / "clientes_segmentados.csv.gz",
        index=False,
        compression="gzip",
    )
    summary.to_csv(ARTIFACTS_DIR / "resumen_segmentos.csv", index=False, encoding="utf-8-sig")
    score_df.to_csv(ARTIFACTS_DIR / "silhouette_kmeans.csv", index=False, encoding="utf-8-sig")
    segment_meta = {
        "k": selected_k,
        "silhouette": float(score_df.loc[score_df["k"].eq(selected_k), "silhouette_score"].iloc[0]),
        "clientes": int(len(clients)),
        "pca_varianza_explicada": [float(v) for v in pca.explained_variance_ratio_],
    }
    return clients, summary, segment_meta


def create_dashboard_data(df: pd.DataFrame, clients: pd.DataFrame) -> None:
    selected_columns = [
        "Fecha",
        "FechaOI",
        "ANIO",
        "numHP",
        "idUnidNeg",
        "DES_AO",
        "Desc_Servicio",
        "NomCliente",
        "duracion",
        "LugarEjec",
        "nParticipante",
        "Consultor",
        "MON",
        "ImporteB",
        "impFacturado",
        "Rubro",
        "Desc_Estado",
        "DescAnulado",
        "propuesta_convertida",
        "dias_conversion_hp_oi",
        "diferencia_facturacion",
        "ingreso_neto_estimado",
        "etapa_comercial",
        "motivo_negocio",
    ]
    dashboard = df[selected_columns].copy()
    dashboard.insert(0, "case_id", dashboard.index.astype(str))
    dashboard = dashboard.merge(
        clients[["NomCliente", "segmento_negocio"]], on="NomCliente", how="left"
    )
    dashboard["segmento_negocio"] = dashboard["segmento_negocio"].fillna("Sin segmento validado")
    dashboard.to_csv(
        ROOT / "data" / "dashboard_data.csv.gz", index=False, compression="gzip"
    )

    cases = dashboard[
        dashboard["ImporteB"].notna()
        & dashboard["MON"].isin(VALID_CURRENCIES)
        & dashboard["Fecha"].notna()
    ].copy()
    cases = cases.sort_values("Fecha", ascending=False).head(2500)
    cases.to_csv(
        ARTIFACTS_DIR / "casos_simulador.csv.gz", index=False, compression="gzip"
    )


def main() -> None:
    print("Cargando base oficial...")
    df = load_and_prepare_data()
    print("Dimensión:", df.shape)

    print("Construyendo indicadores de negocio...")
    business_summary = build_business_summary(df)

    print("Entrenando Random Forest de clasificación pre-OI...")
    classifier, classification_metrics, classification_importance, _ = train_classification(df)
    joblib.dump(classifier, MODELS_DIR / "rf_clasificacion_pre_oi.joblib", compress=3)
    pd.DataFrame([classification_metrics | {"Umbrales": json.dumps(classification_metrics["Umbrales"], ensure_ascii=False), "Matriz_confusion": json.dumps(classification_metrics["Matriz_confusion"]) }]).drop(columns=[]).to_csv(
        ARTIFACTS_DIR / "metricas_clasificacion.csv", index=False, encoding="utf-8-sig"
    )
    classification_importance.to_csv(
        ARTIFACTS_DIR / "importancia_clasificacion.csv", index=False, encoding="utf-8-sig"
    )

    print("Entrenando Random Forest de regresión por moneda...")
    regressors, regression_metrics, regression_importance, _ = train_regression(df)
    for currency, model in regressors.items():
        suffix = "pen" if currency == "S/." else "usd"
        joblib.dump(model, MODELS_DIR / f"rf_regresion_{suffix}.joblib", compress=3)
    regression_metrics.to_csv(
        ARTIFACTS_DIR / "metricas_regresion.csv", index=False, encoding="utf-8-sig"
    )
    regression_importance.to_csv(
        ARTIFACTS_DIR / "importancia_regresion.csv", index=False, encoding="utf-8-sig"
    )

    print("Construyendo segmentación K-Means validada...")
    clients, segment_summary, segment_meta = build_segments(df)
    create_dashboard_data(df, clients)

    metadata = {
        "project": "Eagle Consulting S.A.C. - Dashboard de Machine Learning",
        "generated_from": "Notebook V5 y base 02_1_base_con_nuevos_datos.xlsx",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "valid_currencies": VALID_CURRENCIES,
        "classification_features": {
            "numeric": NUMERIC_CLASS_FEATURES,
            "categorical": CATEGORICAL_CLASS_FEATURES,
        },
        "regression_features": {
            "numeric": NUMERIC_REG_FEATURES,
            "categorical": CATEGORICAL_REG_FEATURES,
        },
        "classification": classification_metrics,
        "regression": regression_metrics.to_dict(orient="records"),
        "segmentation": segment_meta,
        "business_summary": business_summary,
        "limitations": [
            "La probabilidad de clasificación corresponde a facturación alta relativa dentro de cada moneda; no es probabilidad de conversión ni garantía de venta.",
            "La regresión estima el monto en la moneda seleccionada y no mezcla PEN con USD.",
            "Los modelos utilizan patrones históricos y apoyan, pero no reemplazan, el criterio comercial.",
            "El pronóstico temporal mensual no se utiliza de forma operativa porque su desempeño es todavía experimental.",
        ],
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model_use = pd.DataFrame(
        [
            {
                "Modelo": "K-Means",
                "Decisión": "Segmentar la cartera",
                "Uso": "Diferenciar retención/cuentas clave de desarrollo y venta cruzada.",
            },
            {
                "Modelo": "Random Forest clasificación pre-OI",
                "Decisión": "Priorizar oportunidades",
                "Uso": "Ordenar HP por potencial de facturación alta relativa dentro de su moneda.",
            },
            {
                "Modelo": "Random Forest regresión por moneda",
                "Decisión": "Estimar facturación",
                "Uso": "Estimar el monto esperado en PEN o USD sin mezclar unidades.",
            },
            {
                "Modelo": "Modelo temporal mensual",
                "Decisión": "Explorar pronóstico",
                "Uso": "Solo experimental; no automatizar presupuestos con el desempeño actual.",
            },
        ]
    )
    model_use.to_csv(ARTIFACTS_DIR / "matriz_uso_modelos.csv", index=False, encoding="utf-8-sig")

    recommendations = pd.DataFrame(
        [
            {"Prioridad": 1, "Recomendación": "Depurar HP activas antiguas", "Acción": "Revisar propuestas abiertas con más de 90 días, actualizar su estado y cerrar registros no vigentes."},
            {"Prioridad": 2, "Recomendación": "Revisar cotización y precio", "Acción": "Analizar casos de costo de referencia y precio/presupuesto para ajustar rangos, descuentos o argumentos comerciales."},
            {"Prioridad": 3, "Recomendación": "Gestionar cartera por segmento", "Acción": "Retener cuentas de alto valor y desarrollar venta cruzada en clientes de menor actividad."},
            {"Prioridad": 4, "Recomendación": "Usar RF como ranking pre-OI", "Acción": "Priorizar HP de mayor potencial sin automatizar la decisión comercial."},
            {"Prioridad": 5, "Recomendación": "Separar PEN y USD", "Acción": "Mantener estimaciones y reportes monetarios por moneda."},
            {"Prioridad": 6, "Recomendación": "Fortalecer gobierno de datos", "Acción": "Estandarizar fechas, motivos, estados y responsables; excluir identificadores y variables post-resultado."},
        ]
    )
    recommendations.to_csv(ARTIFACTS_DIR / "recomendaciones.csv", index=False, encoding="utf-8-sig")

    print("\nArtefactos generados correctamente.")
    print("Métricas clasificación:", classification_metrics)
    print("Métricas regresión:\n", regression_metrics)
    print("Segmentación:\n", segment_summary)


if __name__ == "__main__":
    main()
