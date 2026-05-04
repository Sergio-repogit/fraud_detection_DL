# Detección de Fraude con Deep Learning

Proyecto de Machine Learning orientado a la detección de transacciones fraudulentas mediante técnicas de Deep Learning y aprendizaje automático supervisado.

##  Descripción del Proyecto

Este proyecto implementa y compara múltiples modelos de detección de fraude sobre un dataset de 5 millones de transacciones financieras. El objetivo es identificar patrones de comportamiento fraudulento manteniendo un equilibrio entre la detección de fraudes (recall) y la minimización de falsos positivos.

### Características del Dataset

- **Tamaño original:** 5.000.000 transacciones
- **Fraudes detectados:** 179.553 (3,6% del total)
- **Periodo temporal:** Enero 2023 - Enero 2024
- **Variables:** 18 características (12 tras preprocesado)

El dataset presenta un desbalanceo significativo entre clases, lo que requiere técnicas especializadas para el entrenamiento efectivo de los modelos.

## Librerías Principales

- **Análisis de datos:** NumPy, Pandas
- **Visualización:** Matplotlib
- **Machine Learning:** Scikit-learn, Imbalanced-learn
- **Deep Learning:** 
  - Keras/TensorFlow
  - PyTorch
  - PyTorch Geometric (GNN)
- **Gradient Boosting:** XGBoost

## Modelos Implementados

1. **Regresión Logística** (baseline)
2. **Support Vector Machines (SVM)**
3. **XGBoost** 
4. **Redes Neuronales Densas (MLP)** con Keras
5. **MLP con Focal Loss** (PyTorch)
6. **Autoencoder** para detección de anomalías
7. **Graph Neural Networks (GNN)**

## Preprocesamiento de Datos

### Reducción del Dataset

Para hacer el dataset manejable sin perder información de fraudes:

```python
# Mantener todos los fraudes + muestra aleatoria de no-fraudes
data reducido: 1.500.000 registros
Fraudes: 179.553 (11,75%)
No fraudes: 1.320.447 (88,25%)
```
Este paso se realiza debido a que nuestro ordenador no puede realizar todo el proceso con los datos completos.

### Feature Engineering

1. **Variables eliminadas:**
   - `transaction_id`, `sender_account`, `receiver_account` (identificadores)
   - `geo_anomaly_score`, `ip_address`, `device_hash` (no relevantes)
   - `fraud_type` (redundante con target y solo posee un tipo de fraude el resto son nan)

2. **Imputación:**
   - `time_since_last_transaction`: NaN a 0 (mediana, media y moda)

3. **Transformaciones:**
   - `timestamp` → datetime + extracción de hora
   - `amount` → `amount_log` (transformación logarítmica) debido a la gran concentración en los valores iniciales.

4. **Encoding:**
   - Variables numéricas: StandardScaler
   - Variables categóricas: OneHotEncoder

### Variables Finales

**Numéricas:**
- `amount_log`
- `time_since_last_transaction`
- `spending_deviation_score`
- `velocity_score`
- `hora` (extraída de timestamp)

**Categóricas:**
- `transaction_type`
- `merchant_category`
- `location`
- `device_used`
- `payment_channel`
- `authentication_method`

## Manejo del Desbalanceo de Clases

Dada la naturaleza desbalanceada del dataset (88% no-fraude vs 12% fraude tras la reduccion de las observaciones), se implementaron múltiples estrategias:

### 1. SMOTE (Synthetic Minority Over-sampling Technique)
Genera ejemplos sintéticos de la clase minoritaria en el espacio de features.

### 2. Class Weights
Penalización diferenciada de errores según la clase:
- **XGBoost:** `scale_pos_weight = 7.35`
- **Keras:** `class_weight = {0: 0.57, 1: 4.18}`

### 3. Focal Loss
Implementada en PyTorch para penalizar más ejemplos difíciles de clasificar:
```python
alpha = 0.75  # Peso para clase positiva
gamma = 2.0   # Factor de enfoque
```

## Metodología

### División de Datos

```python
Train: 80% 
Test:  20% 
Estratificación: Proporciones de clase mantenidasn en train y test
```

### Validación

- **Validación simple:** Train/Test split estratificado
- **Validación cruzada:** 10-fold StratifiedKFold
- **GridSearchCV:** Optimización de hiperparámetros con CV

### Métricas de Evaluación

Dado el desbalanceo, las métricas tradicionales (accuracy) no son representativas. Se priorizan:

- **PR-AUC (Precision-Recall AUC):** Métrica principal
- **ROC-AUC:** Discriminación entre clases
- **Recall (Sensitivity):** Proporción de fraudes detectados
- **Precision:** Proporción de alertas correctas
- **F1-Score:** Balance precision-recall

## Resultados Principales

### Comparativa de Modelos (Test Set)

| Modelo | PR-AUC | ROC-AUC | Recall | Precision | F1-Score |
|--------|---------|---------|--------|-----------|----------|
| Regresión Logística | 0.12 | 0.500 | 0.00 | 0.00  | 0.00  |
| Logística + SMOTE | 0.12 | 0.502 | 0-55 | 0.12 | 0.2 |
| SVM | 0.120 | 0.501 | 1.00 | 0.12 | 0.21 |
| SVM + SMOTE | 0.12 | 0.502 | 0.55 | 0.12 | 0.2 |
| XGBoost | 0.143 | 0.593 | 0.99 | 0.14 | 0.251 |
| **XGBoost + SMOTE** | **0.143** | **0.592** | **1.0** | **0.14** | **0.25** |
| **Deep Learning (Keras)** | **0.144** | **0.593** | **0.96** | **0.14** | **0.25** |
| MLP + Focal Loss | 0.12 | 0.49 | 0.02 | 0.12 | 0.03 |
| Autoencoder | 0.13 | 0.527 | 0.13 | 0.06 | 0.08 |
| GNN | 0.12 | 0.49 | 0.4 | 0.12 | 0.18 |

### Modelos Destacados

#### XGBoost + SMOTE 

```
PR-AUC:  0.143
ROC-AUC: 0.592
Recall:  1.0  
```

**Ventajas:**
- Detecta prácticamente todos los fraudes
- Mejor discriminación entre clases
- Robusto ante desbalanceo

**Desventajas:**
- Alta tasa de falsos positivos (85,7% de alertas son falsas)

####  Deep Learning (Keras)

```
Mejores hiperparámetros:
- batch_size: 1024
- epochs: 30
- optimizer: adam

PR-AUC:  0.144
ROC-AUC: 0.593
Recall:  0.96
```

**Ventajas:**
- Focal Loss maneja bien ejemplos difíciles
- Buen recall manteniendo precision competitiva


### Modelos Experimentales

#### Autoencoder (Detección de Anomalías)

Enfoque no supervisado entrenado solo con transacciones legítimas para deectar las transacciones que no se parezcan a la norma (las fraudulentas)

**Limitaciones:** Baja capacidad discriminativa, alta tasa de falsos positivos.

#### Graph Neural Network (GNN)

Red sobre grafo de transacciones (KNN con k=5):

**Limitaciones:** Restricciones de memoria, rendimiento inferior a modelos tradicionales.


### Instalación de Dependencias

```bash
# Librerías base
pip install numpy pandas matplotlib scikit-learn

# Manejo de desbalanceo
pip install imbalanced-learn

# Modelos avanzados
pip install xgboost scikeras

# Deep Learning
pip install tensorflow keras torch torch-geometric
```

### Ejecución

```bash
# Clonar repositorio
git clone https://github.com/usuario/deteccion-fraude-deep-learning.git
cd deteccion-fraude-deep-learning

# Ejecutar notebook
jupyter notebook codigo_proyecto_final_ML.ipynb
```

### Dataset

El dataset `financial_fraud_detection_dataset.csv` debe colocarse en el mismo directorio que el notebook.

**Nota:** Debido al tamaño (5M registros), el código incluye reducción automática del dataset.

## Estructura del Proyecto

```
deteccion-fraude-deep-learning/
│
├── codigo_proyecto_final_ML.ipynb   # Notebook principal
├── README.md                         # Este archivo
├── RESULTADOS.md                     # Análisis detallado de resultados
└── financial_fraud_detection_dataset.csv  # Dataset 
```

## Conclusiones

1. **XGBoost y la red neuronal optimizada demuestran ser los modelos más efectivos** para este problema.

2. **El manejo del desbalanceo es crítico:** Los modelos sin técnicas específicas (SMOTE, class weights, focal loss) colapsan hacia la clase mayoritaria.

3. **Trade-off precision-recall:** Todos los modelos mantienen precision baja (12-15%) debido al desbalanceo extremo. En producción, se requeriría un sistema de verificación secundaria.

4. **Deep Learning competitivo pero no superior:** Las redes neuronales alcanzan rendimiento similar a XGBoost pero con mayor complejidad computacional.

5. **Modelos experimentales limitados:** Autoencoders y GNNs no superan a métodos tradicionales en este caso específico.

## Trabajo Futuro

- **Ensemble methods:** Combinar predicciones de múltiples modelos
- **Feature engineering avanzado:** Agregaciones temporales, patrones de comportamiento
- **Ajuste de umbrales:** Optimizar punto de corte según coste de negocio
- **Modelos específicos de serie temporal:** Capturar dependencias temporales
- **Análisis de costes:** Integrar costes de FP vs FN en la optimización
- **Mejora de hardware**: para poder hacer el estudio con la base completa

## Autores

Sergio Mínguez Cruces
Diego Vega Stergar

## Licencia

Este proyecto es de código abierto y está disponible para fines educativos.
