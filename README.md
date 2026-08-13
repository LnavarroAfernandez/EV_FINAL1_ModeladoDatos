# Eagle Intelligence — Dashboard de Machine Learning con Plotly Dash

Aplicación web interactiva adaptada al proyecto **Modelado y Presentación — Eagle Consulting S.A.C.**. El dashboard transforma los resultados del Notebook V5 en una herramienta comprensible para una audiencia comercial, financiera o directiva.

## GRUPO 4 INTEGRANTES
- Christian Alejandro Muro Chaname
- Katia Lisbeth Francia Ugaz
- Ivo Boris Arias Suni
- Nicolas Junior Samaniego Cardenas
- Luis Angel Navarro Fernandez

## Qué incluye

### 1. Resumen comercial

- Filtros por periodo y área de servicio.
- Embudo histórico HP → OI.
- Conversión por línea de servicio.
- HP activas e inactivas sin OI.
- Mediana de días de conversión.
- Facturación PEN y USD mostrada por separado.
- Principales motivos de no conversión.
- Evolución anual de registros y conversión.

### 2. Segmentación de clientes

- K-Means validado con `k = 2`.
- Perfil **Menor actividad y alta conversión**.
- Perfil **Alto valor y alta actividad**.
- Peso de cada segmento en clientes, operaciones y facturación.
- Top de clientes por segmento y moneda.
- Evidencia técnica PCA en un bloque desplegable.

### 3. Simulador pre-OI

Adapta los ejercicios de la clase con:

- `Dropdown` para seleccionar una HP histórica.
- `Slider` para simular cambios en importe, participantes y duración.
- Botón que actúa como `Input`; los controles actúan como `State`.
- Gauge con la probabilidad de **facturación alta relativa dentro de la moneda**.
- Estimación de facturación mediante Random Forest de regresión.
- Rango de referencia P10–P90 a partir de los árboles del bosque.
- Umbral de facturación alta calculado únicamente con el conjunto de entrenamiento.

> La probabilidad mostrada no es probabilidad de conversión ni garantía de venta. Es un ranking relativo de potencial de facturación dentro de PEN o USD.

### 4. Validación de modelos

- Accuracy, precisión, recall, F1 y ROC-AUC del Random Forest pre-OI.
- Matriz de confusión del modelo validado.
- Importancia de variables traducida a lenguaje de negocio.
- Evaluación de regresión por moneda.
- Gráfico real frente a estimado.
- Filtros que revisan el comportamiento sobre la muestra de prueba sin reentrenar el modelo.
- Matriz modelo → decisión de negocio.

### 5. Conclusiones y recomendaciones

- Depuración de HP activas antiguas.
- Revisión de precio y cotización.
- Gestión diferenciada por segmento.
- Uso de Random Forest como ranking pre-OI.
- Separación estricta de PEN y USD.
- Fortalecimiento del gobierno de datos.
- Advertencia sobre el modelo temporal mensual experimental.


## Estructura del proyecto

```text
eagle_dashboard_plotly_dash/
├── app.py
├── requirements.txt
├── README.md
├── run_dashboard.bat
├── run_dashboard.sh
├── assets/
│   └── style.css
├── data/
│   ├── 02_1_base_con_nuevos_datos.xlsx
│   └── dashboard_data.csv.gz
├── models/
│   ├── rf_clasificacion_pre_oi.joblib
│   ├── rf_regresion_pen.joblib
│   └── rf_regresion_usd.joblib
├── artifacts/
│   ├── metadata.json
│   ├── resumen_segmentos.csv
│   ├── clientes_segmentados.csv.gz
│   ├── metricas_regresion.csv
│   ├── importancia_clasificacion.csv
│   ├── importancia_regresion.csv
│   └── otros archivos de soporte
├── scripts/
│   └── prepare_artifacts.py
├── src/
│   ├── services.py
│   └── figures.py
└── tests/
    └── smoke_test.py
```

## Cómo ejecutar en Windows

### 1. Clonar el repositorio y abrir en Visual Studio Code

Ejecute el siguiente comando en su terminal para descargar el proyecto desde GitHub:

```powershell
1. git clone https://github.com/LnavarroAfernandez/EV_FINAL1_ModeladoDatos.git
```

### 2. Abrir la carpeta del proyecto en Visual Studio Code

Ubíquese dentro de la carpeta `EV_FINAL1_ModeladoDatos`.

### 3. Crear un entorno virtual

```powershell
2. cd EV_FINAL1_ModeladoDatos
3. python -m venv .venv
4. .venv\Scripts\activate
```

### 4. Instalar dependencias

```powershell
5. python -m pip install --upgrade pip
6. pip install -r requirements.txt
```

### 5. Ejecutar la aplicación

```powershell
7. python app.py
```

Abrir en el navegador:

```text
8. http://127.0.0.1:8050
```

También puede usar:

```powershell
run_dashboard.bat
```

## Cómo ejecutar en Linux o macOS

```bash
git clone [https://github.com/LnavarroAfernandez/EV_FINAL1_ModeladoDatos.git](https://github.com/LnavarroAfernandez/EV_FINAL1_ModeladoDatos.git)
cd EV_FINAL1_ModeladoDatos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
O:

```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

## Verificación rápida

Antes de abrir Dash puede ejecutar:

```bash
python tests/smoke_test.py
```

La prueba comprueba:

- Lectura de la base procesada.
- Carga de los tres modelos.
- Predicción de un escenario.
- Generación de figuras Plotly.
- Coherencia de métricas y artefactos.

## Regenerar los modelos y archivos auxiliares

Los modelos y artefactos ya están incluidos. Solo es necesario regenerarlos si cambia la base o si existe incompatibilidad entre versiones de Scikit-learn:

```bash
python scripts/prepare_artifacts.py
```

El script trabaja con rutas relativas y vuelve a crear:

- Modelos de clasificación y regresión.
- Métricas.
- Predicciones de prueba.
- Segmentación K-Means.
- Tablas para el dashboard.

## Resultados técnicos incorporados

### Clasificación Random Forest pre-OI

- Accuracy aproximada: **91.25%**.
- Precisión aproximada: **92.92%**.
- Recall aproximado: **88.01%**.
- F1 aproximado: **90.40%**.
- ROC-AUC aproximado: **96.66%**.
- Umbral de facturación alta PEN: **S/ 3,000**.
- Umbral de facturación alta USD: **US$ 2,754**.

### Regresión por moneda

| Moneda | Registros | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| PEN | 11,288 | S/ 3,170.73 | S/ 30,920.27 | 0.7505 |
| USD | 650 | US$ 3,681.62 | US$ 16,881.54 | 0.5670 |

### K-Means

- Clientes segmentados: **1,320**.
- `k = 2`.
- Silhouette validado: **0.3845**.
- 974 clientes de menor actividad y alta conversión.
- 346 clientes de alto valor y alta actividad.

## Decisiones metodológicas respetadas

- No se mezcla PEN con USD.
- `idAreaCosto` no se usa como factor de negocio.
- Se excluyen variables posteriores a la OI del modelo pre-OI.
- La importancia de variables no se presenta como causalidad.
- Los modelos apoyan la decisión comercial, pero no la automatizan.
- El modelo temporal mensual se mantiene como experimental.

## Solución de problemas

### El comando `python app.py` no encuentra archivos

Ejecute el comando desde la carpeta raíz del proyecto, donde se encuentran `app.py`, `data/`, `models/` y `artifacts/`.

### Error al cargar un modelo Joblib

Instale la versión indicada de Scikit-learn:

```bash
pip install scikit-learn==1.8.0
```

También puede regenerar los modelos:

```bash
python scripts/prepare_artifacts.py
```

### Puerto 8050 ocupado

Detenga la aplicación anterior con `Ctrl + C` o ejecute con otro puerto:

```powershell
set PORT=8051
python app.py
```

En Linux/macOS:

```bash
PORT=8051 python app.py
```

## Fuente de datos y alcance

- Base oficial enriquecida: `02_1_base_con_nuevos_datos.xlsx`.
- Fuente técnica: Notebook V5 del proyecto.
- Periodo histórico observado: 2012–2026.
- El dashboard utiliza patrones históricos y debe complementarse con información operativa no contenida en la base.
