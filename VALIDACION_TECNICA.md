# Validación técnica del dashboard

## Pruebas realizadas

- La base procesada se carga correctamente: **17,962 registros**.
- Los tres modelos Joblib se cargan correctamente.
- El simulador genera una probabilidad válida entre 0 y 1.
- La regresión devuelve una estimación no negativa y un rango P10–P90 coherente.
- Las figuras Plotly se generan sin errores:
  - embudo HP → OI;
  - motivos de no conversión;
  - evolución anual;
  - participación de segmentos;
  - PCA;
  - gauge;
  - importancias;
  - real frente a estimado.
- Los callbacks fueron importados y ejecutados con una capa de prueba local:
  - resumen comercial;
  - top de clientes;
  - selección de caso;
  - simulador;
  - validación de regresión.
- Se verificó que las rutas del proyecto sean relativas.

## Resultado del smoke test

```text
SMOKE TEST OK
- Registros: 17,962
- ROC-AUC clasificación: 0.9666
- K-Means: k=2 · silhouette=0.3845
- Regresión PEN: R²=0.7505 · RMSE=30,920.27
```
