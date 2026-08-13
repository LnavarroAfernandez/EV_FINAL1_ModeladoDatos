from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.figures import (  # noqa: E402
    annual_trend_figure,
    feature_importance_figure,
    funnel_figure,
    pca_figure,
    probability_gauge,
    reasons_figure,
    regression_scatter_figure,
    segment_share_figure,
)
from src.services import (  # noqa: E402
    filter_dashboard_data,
    load_classification_importance,
    load_dashboard_data,
    load_metadata,
    load_regression_importance,
    load_segment_summary,
    load_segmented_clients,
    predict_scenario,
    regression_subset_metrics,
)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    metadata = load_metadata()
    data = load_dashboard_data()

    assert_true(len(data) == 17_962, "La base del dashboard no tiene 17,962 registros.")
    assert_true(metadata["classification"]["ROC_AUC"] > 0.95, "ROC-AUC de clasificación inesperado.")
    assert_true(metadata["segmentation"]["k"] == 2, "K-Means no conserva k=2.")

    result = predict_scenario(None, 10, 20, 0)
    assert_true(0 <= result["probability"] <= 1, "Probabilidad fuera de rango.")
    assert_true(result["estimate"] >= 0, "Estimación negativa.")
    assert_true(result["range_low"] <= result["range_high"], "Rango de referencia inválido.")

    filtered = filter_dashboard_data([2012, 2026], "Todas")
    figures = [
        funnel_figure(filtered),
        reasons_figure(filtered),
        annual_trend_figure(filtered),
        segment_share_figure(load_segment_summary()),
        pca_figure(load_segmented_clients()),
        probability_gauge(result["probability"], result["level"]),
        feature_importance_figure(
            load_classification_importance(),
            "Importancia de clasificación",
        ),
    ]
    assert_true(all(len(fig.data) > 0 for fig in figures), "Una figura Plotly quedó vacía.")

    reg_subset, reg_metrics = regression_subset_metrics("S/.", [2012, 2026], "Todas")
    assert_true(reg_metrics["records"] > 2_000, "Muestra de regresión PEN inesperadamente pequeña.")
    assert_true(reg_metrics["r2"] is not None and reg_metrics["r2"] > 0.70, "R² PEN inesperado.")
    reg_fig = regression_scatter_figure(reg_subset, "S/.")
    reg_imp_fig = feature_importance_figure(
        load_regression_importance().query("Moneda == 'S/.'"),
        "Importancia de regresión PEN",
    )
    assert_true(len(reg_fig.data) == 2, "El gráfico real vs. estimado debe tener puntos y diagonal.")
    assert_true(len(reg_imp_fig.data) == 1, "La importancia de regresión no se generó.")

    print("SMOKE TEST OK")
    print(f"- Registros: {len(data):,}")
    print(f"- ROC-AUC clasificación: {metadata['classification']['ROC_AUC']:.4f}")
    print(f"- K-Means: k={metadata['segmentation']['k']} · silhouette={metadata['segmentation']['silhouette']:.4f}")
    print(f"- Predicción de prueba: {result['probability']:.2%} · {result['estimate']:,.2f} {result['currency']}")
    print(f"- Regresión PEN: R²={reg_metrics['r2']:.4f} · RMSE={reg_metrics['rmse']:,.2f}")


if __name__ == "__main__":
    main()
