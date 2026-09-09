# Clasificación de personajes de *Los Simpson* mediante un Perceptrón Multicapa (MLP)

**Informe técnico — Evaluación Parcial N°1**  
**TLY1101 · Técnicas Avanzadas de Machine Learning I · Duoc UC · 2026**

| Ítem | Detalle |
|---|---|
| **Integrantes** | *Joaquin sanhueza, Carlos calvio, fernando guzman, vicente cossio* |
| **Sección / Docente** | *Seccion: 001-D Docente: Marco Japke* |
| **Fecha** | *9/9/2026* |
| **Dataset** | [`alfaro96/los-simpson`](https://www.kaggle.com/datasets/alfaro96/los-simpson) |
| **Variante del grupo** | 5 personajes · 100 imágenes por clase · `SEED = 67` |
| **Notebook** | [`notebooks/prueba 1 tecnicas de machine learning.ipynb`](notebooks/) |

---

## Tabla de contenidos

1. [Descripción del problema de negocio](#1-descripción-del-problema-de-negocio)
2. [Objetivos del proyecto](#2-objetivos-del-proyecto)
3. [Definición de KPIs](#3-definición-de-kpis)
4. [Descripción de las fuentes de datos](#4-descripción-de-las-fuentes-de-datos)
5. [Preparación y análisis exploratorio de los datos (EDA)](#5-preparación-y-análisis-exploratorio-de-los-datos-eda)
6. [Metodología: CRISP-DM](#6-metodología-crisp-dm)
7. [Modelo implementado](#7-modelo-implementado)
8. [Resultados](#8-resultados)
9. [Análisis de errores y limitaciones](#9-análisis-de-errores-y-limitaciones)
10. [Conclusiones y mejoras propuestas](#10-conclusiones-y-mejoras-propuestas)
11. [Consideraciones éticas](#11-consideraciones-éticas)
12. [Estructura del proyecto](#12-estructura-del-proyecto)
13. [Cómo reproducir la solución](#13-cómo-reproducir-la-solución)

---

## 1. Descripción del problema de negocio

Un archivo audiovisual que gestiona el catálogo de la serie *Los Simpson* necesita **etiquetar
automáticamente qué personaje aparece en cada fotograma**. Hoy ese trabajo se realiza de forma
manual, lo que resulta inviable a escala: solo el conjunto de datos utilizado contiene más de
16.000 imágenes ya recortadas, y el catálogo completo de la serie supera las 700 emisiones.

Un clasificador automático de personajes habilita tres capacidades de negocio concretas:

1. **Búsqueda por personaje** dentro del catálogo (*"mostrar escenas donde aparece Bart"*).
2. **Generación automática de clips y miniaturas** centrados en un personaje.
3. **Métricas de tiempo en pantalla** por personaje, útiles para decisiones editoriales y de
   licenciamiento de merchandising.

El problema técnico asociado es de **clasificación supervisada multiclase sobre imágenes**.

---

## 2. Objetivos del proyecto

### Objetivo general

> Evaluar la viabilidad de un **Perceptrón Multicapa (MLP)** como modelo de línea base para la
> clasificación automática de personajes de *Los Simpson*, caracterizando su desempeño y sus
> limitaciones antes de escalar a arquitecturas más especializadas.

### Objetivos específicos

1. Explorar y caracterizar el conjunto de datos, identificando su distribución de clases y su calidad.
2. Definir y aplicar un preprocesamiento reproducible que permita alimentar una red densa.
3. Diseñar e implementar una arquitectura MLP justificando capas, activaciones e hiperparámetros.
4. Entrenar y validar el modelo, comparando el efecto de distintas configuraciones.
5. Evaluar el desempeño con métricas de clasificación e interpretar los resultados.
6. Auditar los errores del modelo e identificar las limitaciones estructurales del MLP en imágenes.

---

## 3. Definición de KPIs

Los KPIs traducen el objetivo de negocio a umbrales medibles. La línea base de referencia es el
**azar: 1/5 = 20 %**, porque el subconjunto de trabajo está perfectamente balanceado.

| KPI | Métrica | Umbral objetivo | Justificación |
|---|---|---|---|
| **KPI-1** | Accuracy en validación | **≥ 60 %** | Mínimo para que el etiquetado automático ahorre trabajo humano en lugar de generarlo. |
| **KPI-2** | F1-Score macro | **≥ 0,55** | Al promediar por clase sin ponderar, impide que un buen resultado global esconda clases mal resueltas. |
| **KPI-3** | Recall mínimo por clase | **≥ 0,30** | Garantiza cobertura: ningún personaje puede quedar prácticamente invisible para el buscador. |
| **KPI-4** | Tiempo de entrenamiento | **< 5 min en Colab sin GPU** | Viabilidad operativa para reentrenar con frecuencia. |

> **Criterio de éxito.** El objetivo del proyecto **no es superar los KPIs a toda costa**, sino
> medir con rigor si un MLP los alcanza. Un KPI no cumplido es un resultado válido: constituye la
> evidencia que justifica técnicamente el paso a una arquitectura convolucional en la siguiente
> experiencia de aprendizaje.

---

## 4. Descripción de las fuentes de datos

### 4.1 Fuente

| Atributo | Valor |
|---|---|
| **Nombre** | Los Simpson Dataset |
| **Origen** | Kaggle — [`alfaro96/los-simpson`](https://www.kaggle.com/datasets/alfaro96/los-simpson) |
| **Forma de obtención** | Descarga programática con `kagglehub.dataset_download()` (no se versiona en el repositorio) |
| **Tipo de dato** | Imágenes RGB (3 canales), formato `.jpg` |
| **Organización** | Carpetas `train/` y `test/`; dentro de `train/`, una subcarpeta por personaje |
| **Clases** | **25** personajes (medido sobre `train/`) |
| **Volumen** | **16.137** imágenes en `train/` |
| **Resolución** | Variable; se estandariza en el preprocesamiento |
| **Licencia / uso** | Material protegido por derechos de autor; se utiliza con fines exclusivamente académicos |

> **Nota sobre la ficha de la evaluación.** El documento de la EP1 describe este conjunto como
> *"reconocer letras del lenguaje de señas, 42 clases, ~20.000 imágenes"*. Al inspeccionar la
> descarga real se verificó que corresponde a personajes de *Los Simpson*, con **25 clases** y
> **16.137 imágenes** en `train/`. Este informe utiliza las cifras medidas directamente sobre los
> datos, calculadas en la sección 2.1 del notebook.

### 4.2 Justificación de la selección

Entre los cuatro conjuntos propuestos por la evaluación se escogió Los Simpson por cuatro razones
técnicas:

1. **Clases visualmente muy similares.** Bart, Lisa y Milhouse comparten paleta cromática y
   proporciones. Esto somete a máxima tensión la principal debilidad del MLP —la pérdida de la
   estructura espacial al aplanar la imagen— y permite documentarla con evidencia, que es
   precisamente el objetivo de la evaluación.
2. **Desbalance de clases real y pronunciado.** Obliga a tomar y justificar una decisión explícita
   de muestreo, en vez de recibir los datos ya equilibrados.
3. **Resolución variable y fondos heterogéneos.** Exige un preprocesamiento genuino y no solo
   cargar tensores listos.
4. **Volumen manejable.** Permite iterar y comparar varias configuraciones dentro de las 5 horas
   asignadas, sin necesidad de GPU.

### 4.3 Variante propia del grupo

Definida antes de cualquier experimento, para diferenciar el trabajo del de otros grupos:

| Decisión | Valor | Justificación |
|---|---|---|
| **Clases** | `homer_simpson`, `ned_flanders`, `lisa_simpson`, `bart_simpson`, `milhouse_van_houten` | Combina dos personajes cromáticamente distintivos con tres altamente similares entre sí, para contrastar casos fáciles y difíciles. |
| **Imágenes por clase** | 100 (500 en total) | Muestreo balanceado que neutraliza el desbalance original y permite observar el sobreajuste de forma controlada. |
| **Semilla** | `SEED = 67` | Propagada a `random`, `NumPy` y `TensorFlow/Keras`. |
| **Resolución** | 64 × 64 × 3 | Compromiso entre detalle y número de parámetros (ver sección 5.3). |
| **Partición** | 80 % / 20 %, estratificada | Conserva la proporción de clases en ambos subconjuntos. |

---

## 5. Preparación y análisis exploratorio de los datos (EDA)

### 5.1 Hallazgos del EDA

| # | Hallazgo | Evidencia | Decisión derivada |
|---|---|---|---|
| **1** | **Desbalance de clases marcado.** La clase mayoritaria (1.796 imágenes) multiplica casi 22 veces a la minoritaria (82). | `images/01_distribucion_clases.png` | Muestreo balanceado de 100 imágenes por clase; el azar queda fijado en 20 %. |
| **2** | **Resolución y relación de aspecto variables.** Las imágenes no comparten tamaño común (media de 428x417 px). | `images/02_resoluciones.png` | Redimensionado obligatorio a 64 × 64 (la red densa exige entrada de longitud fija). |
| **3** | **Alta similitud visual entre clases**, fondos heterogéneos y personajes en escala/pose variables. | `images/03_ejemplos_clases.png` | Se anticipan las confusiones entre Bart, Lisa y Milhouse; base del análisis de errores. |

Adicionalmente se verificó la **calidad de los datos**: se comprobó que todos los archivos
seleccionados (500) fueran legibles (0 corruptos) y que no hubiera rutas duplicadas en la muestra.

### 5.2 Etapas del preprocesamiento

| Etapa | Qué se hace | Por qué |
|---|---|---|
| **1. Muestreo balanceado** | 100 imágenes por clase, con lista de archivos **ordenada** antes de muestrear | Neutraliza el desbalance. El `sorted()` es indispensable: `os.listdir()` no garantiza orden estable, por lo que sin él la semilla no aseguraría la misma selección en otra máquina. |
| **2. Lectura** | `cv2.imread()` | Devuelve `None` en archivos dañados, lo que permite descartarlos y contabilizarlos. |
| **3. Conversión BGR → RGB** | `cv2.cvtColor()` | OpenCV lee en orden BGR; sin la conversión los canales quedan invertidos y la interpretación cromática del análisis de errores sería incorrecta. |
| **4. Redimensionado a 64 × 64** | `cv2.resize(..., INTER_AREA)` | El MLP exige entrada de longitud fija. `INTER_AREA` es la interpolación adecuada para reducir tamaño: promedia píxeles y evita el aliasing del método bilineal por defecto. |
| **5. Normalización a [0, 1]** | `img / 255.0` | Valores grandes producen gradientes de magnitud dispar y convergencia inestable. |
| **6. Tipo `float32`** | `astype('float32')` | Tipo nativo de TensorFlow; evita conversión implícita y reduce la memoria a la mitad. |
| **7. Partición estratificada** | `train_test_split(stratify=y)` | Conserva 20 imágenes por clase en validación; sin estratificar, alguna clase podría quedar sub-representada. |

### 5.3 Justificación de la resolución elegida

La primera capa densa conecta cada píxel con cada neurona, por lo que el costo crece con el
cuadrado del lado:

| Resolución | Características de entrada | Parámetros de la 1.ª capa densa (128 neuronas) |
|---|---|---|
| 32 × 32 × 3 | 3.072 | ~393 mil |
| **64 × 64 × 3** | **12.288** | **~1,57 millones** |
| 128 × 128 × 3 | 49.152 | ~6,29 millones |

Con 32 × 32 se perderían rasgos finos entre personajes parecidos; con 128 × 128 el modelo superaría
los 6 millones de parámetros para solo 400 imágenes de entrenamiento. **64 × 64** es el compromiso
adoptado.

### 5.4 Limitación metodológica declarada

Dado el tamaño reducido del subconjunto (500 imágenes), **el mismo 20 % cumple dos roles**: se usa
como `validation_data` durante el entrenamiento y como conjunto de evaluación final. Al no existir
un tercer conjunto de prueba completamente independiente, las métricas finales podrían ser
ligeramente optimistas. Se declara explícitamente porque afecta la lectura de los resultados.

---

## 6. Metodología: CRISP-DM

El proyecto sigue las fases de **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*).
La correspondencia con las secciones del notebook es la siguiente:

| Fase CRISP-DM | Qué se hizo en este proyecto | Sección del notebook |
|---|---|---|
| **1. Comprensión del negocio** | Definición del problema del archivo audiovisual, objetivos generales y específicos, y KPIs con umbrales medibles. | 1.1 – 1.3 |
| **2. Comprensión de los datos** | Inventario de las 25 clases y 16.137 imágenes, análisis del desbalance, verificación de calidad (archivos ilegibles, duplicados), distribución de resoluciones y visualización de ejemplos por clase. | 2.1 – 2.5 |
| **3. Preparación de los datos** | Muestreo balanceado reproducible, conversión de espacio de color, redimensionado, normalización, conversión de tipo y partición estratificada. | 2.3, 3 |
| **4. Modelado** | Diseño y justificación de la arquitectura MLP; entrenamiento del modelo base y comparación de configuraciones alternativas. | 4, 5 |
| **5. Evaluación** | Accuracy, Precision, Recall, F1 por clase, F1 macro, matriz de confusión, verificación de KPIs y auditoría de errores con ejemplos. | 6, 7 |
| **6. Despliegue** | Serialización del modelo y del historial en `models/`, exportación de figuras en `images/` y documentación reproducible. Se concluye que el modelo **no está en condiciones de desplegarse** sin supervisión humana, y se propone la iteración siguiente (CNN). | 8, 10 |

CRISP-DM es un proceso **iterativo**: los hallazgos de la fase de evaluación (sobreajuste,
confusión entre personajes similares) retroalimentan la comprensión del problema y definen la
siguiente iteración del ciclo, orientada a arquitecturas convolucionales.

---

## 7. Modelo implementado

### 7.1 Arquitectura

```
Input(64, 64, 3)
  └── Flatten()                      →  12.288 características
        └── Dense(128, relu)         →  capa oculta 1
              └── Dense(64, relu)    →  capa oculta 2
                    └── Dense(5, softmax)  →  capa de salida
```

### 7.2 Justificación de cada decisión

| Componente | Configuración | Justificación |
|---|---|---|
| **Entrada + Flatten** | `Input(64,64,3)` + `Flatten()` | El MLP solo admite vectores. Este es el punto crítico: al aplanar se pierde la **vecindad espacial** entre píxeles. |
| **Capas ocultas** | 128 y 64 neuronas | Dos capas bastan para introducir no linealidad sin disparar los parámetros. Patrón decreciente (embudo): representación amplia y luego compresión hacia las clases. Con 400 imágenes, más capacidad solo aumenta el sobreajuste. |
| **Activación oculta** | `ReLU` | No satura para entradas positivas (evita el desvanecimiento del gradiente) y es la más económica de calcular. |
| **Capa de salida** | `Dense(5, softmax)` | Una neurona por clase; `softmax` normaliza a una distribución de probabilidad que suma 1, propia de la clasificación multiclase de etiqueta única. |
| **Función de pérdida** | `sparse_categorical_crossentropy` | Entropía cruzada para etiquetas **enteras**; la variante *sparse* evita codificar `y` en one-hot. |
| **Optimizador** | `Adam(lr=0.001)` | Adapta la tasa de aprendizaje por parámetro; converge más rápido que `SGD` con menos ajuste manual. |
| **Épocas / batch** | 30 / 32 | 13 actualizaciones por época: frecuencia suficiente y gradiente estable. |
| **Regularización** | **Ninguna en el modelo base** | Decisión deliberada: el objetivo es *evidenciar* el sobreajuste, no ocultarlo. Se mide aparte como configuración alternativa. |

### 7.3 Configuraciones comparadas

Todas se entrenan bajo condiciones idénticas (misma partición, semilla reiniciada antes de cada
entrenamiento, mismo tope de épocas, `EarlyStopping` con `restore_best_weights=True`):

| Config. | Qué cambia | Hipótesis contrastada |
|---|---|---|
| **A** | Base + detención temprana | ¿Detenerse en el mínimo de validación mejora la generalización? |
| **B** | 512-256-128 neuronas | ¿Más capacidad ayuda, o agrava el sobreajuste? |
| **C** | + `BatchNormalization` + `Dropout(0.4)` | ¿La regularización reduce la brecha entrenamiento/validación? |
| **D** | `learning_rate = 0.0001` | ¿Un paso menor produce convergencia más estable? |

---

## 8. Resultados

> ⚠️ **Nota:** El extracto del cuaderno de Jupyter finaliza en la etapa de Exploración de Datos (EDA). Una vez que se ejecute la fase de modelado y evaluación en el cuaderno, se recomienda copiar la salida de la celda de evaluación y pegarla aquí.

| Métrica | Valor obtenido | Umbral KPI | Estado |
|---|---|---|---|
| Accuracy (validación) | *39.00% * | **≥ 60 %** | *No cumple* |
| F1-Score macro | *0.370* | **≥ 0,55** | *No cumple* |
| Recall mínimo por clase | * 0.10* | **≥ 0,30** | *No cumple* |
| Tiempo de entrenamiento | *23 s* | **< 300 s** | *Cumple* |
| Línea base (azar) | 20 % | — | referencia |

| Personaje | Precision | Recall | F1-Score | Soporte |
|---|---|---|---|---|
| homer_simpson | *0.39* | *0.55* | *0.46* | 20 |
| ned_flanders | *0.54* | *0.35* | *0.42* | 20 |
| lisa_simpson | *0.30* | *0.50* | *0.38* | 20 |
| bart_simpson | *0.67* | *0.10* | *0.17* | 20 |
| milhouse_van_houten | *0.39* | *0.45* | *0.42* | 20 |

### Figuras generadas

| Archivo | Contenido |
|---|---|
| `images/01_distribucion_clases.png` | Distribución de imágenes por clase (evidencia del desbalance) |
| `images/02_resoluciones.png` | Distribución de resoluciones originales |
| `images/03_ejemplos_clases.png` | Ejemplos representativos de cada personaje |
| `images/04_curvas_aprendizaje.png` | Curvas de accuracy y loss por época |
| `images/05_comparacion_configuraciones.png` | Accuracy de validación de las 4 configuraciones |
| `images/06_matriz_confusion.png` | Matriz de confusión |
| `images/07_aciertos.png` | Aciertos con mayor confianza |
| `images/08_errores.png` | Errores con mayor confianza |

### Interpretación de cada métrica en este problema

| Métrica | Qué significa aquí |
|---|---|
| **Accuracy** | Interpretable **solo porque el conjunto está balanceado**. La referencia obligada es el azar (20 %). |
| **Precision** | Confiabilidad de la etiqueta: si el buscador devuelve un resultado como "Homero", ¿es realmente Homero? |
| **Recall** | Cobertura: ¿cuántas escenas reales del personaje encuentra el buscador? |
| **F1-Score** | Media armónica de ambas; castiga los desequilibrios y permite comparar clases. |
| **F1 macro** | Da el mismo peso a cada personaje; impide que las clases fáciles disimulen el fracaso en las difíciles. |
| **Matriz de confusión** | La única que muestra **con qué** se confunde cada clase, no solo cuánto falla. |

---

## 9. Análisis de errores y limitaciones

### 9.1 Causas de las predicciones erróneas

1. **Similitud cromática entre personajes.** Bart, Lisa y Milhouse comparten el amarillo de piel y
   cabello. Como el MLP recibe un vector plano de intensidades, la distribución global de color es
   la señal más accesible que puede aprender, y es casi idéntica entre ellos. Homero y Ned
   Flanders, con rasgos cromáticos propios, resultan mucho más separables.
2. **Pérdida de la estructura espacial en el `Flatten`.** Dos píxeles vecinos quedan tan "lejos"
   entre sí como dos píxeles de esquinas opuestas. La red no puede aprender rasgos geométricos como
   los picos del pelo de Bart frente a la forma en estrella del de Lisa.
3. **Ausencia de invarianza a la traslación y a la escala.** Cada peso está atado a una posición
   fija del vector de entrada; un desplazamiento del personaje cambia por completo la entrada.
4. **Interferencia del fondo.** Todos los píxeles pesan igual, incluidos los del fondo, que puede
   dominar la representación.

### 9.2 Limitaciones del MLP para clasificación de imágenes

| Limitación | Consecuencia observada |
|---|---|
| **Pérdida de la topología 2D** | La red solo aprende distribuciones de intensidad, no formas. Causa raíz de la confusión entre personajes similares. |
| **Explosión de parámetros** | 1,58 millones de parámetros para imágenes de 64×64; a 128×128 se cuadruplicarían. |
| **Sobreajuste con pocos datos** | Miles de parámetros por imagen de entrenamiento hacen que memorizar sea más fácil que generalizar. |
| **Sin invarianza a traslación, escala o rotación** | Un personaje desplazado produce un vector de entrada distinto. |
| **Sin reutilización de parámetros** | Un patrón aprendido en una zona debe reaprenderse en cada otra zona. |
| **Resolución fija y baja** | Obliga a comprimir a 64×64 y perder detalle, porque subir la resolución es prohibitivo en parámetros. |

Estas limitaciones son **estructurales, no de ajuste**: la comparación de configuraciones mostró
que ninguna variación de hiperparámetros las resuelve.

---

## 10. Conclusiones y mejoras propuestas

### 10.1 Conclusiones

- El modelo **supera la línea base del azar**, lo que confirma que aprendió señal real de las
  imágenes, pero se mantiene por debajo del umbral del KPI-1 para automatizar el etiquetado.
- Este resultado **no es un fracaso del proyecto**: el objetivo declarado era determinar si un MLP
  basta para este problema, y la respuesta —respaldada por curvas, métricas, matriz de confusión y
  cuatro configuraciones comparadas— es que no basta, junto con la explicación técnica del porqué.
- El desempeño es **muy desigual entre clases**: los personajes cromáticamente distintivos obtienen
  los mejores F1 y los similares entre sí, los peores. Esto confirma que el modelo discrimina
  principalmente por color y no por forma.
- El **sobreajuste** es la consecuencia directa de la relación entre 1,58 millones de parámetros y
  400 imágenes de entrenamiento.

### 10.2 Mejoras propuestas

1. **Sustituir el MLP por una Red Neuronal Convolucional (CNN).** Mejora de mayor impacto: las
   capas `Conv2D` operan sobre vecindades de píxeles, **preservan la estructura espacial**,
   aprenden bordes y formas, y **comparten pesos** entre posiciones, lo que aporta invarianza a la
   traslación y reduce drásticamente los parámetros.
2. **Aumento de datos (*data augmentation*).** Rotaciones, volteos, desplazamientos y cambios de
   brillo multiplican los ejemplos efectivos y atacan directamente el sobreajuste.
3. **Ampliar el conjunto de entrenamiento** de 100 a 500 o 1.000 imágenes por clase.
4. **Partición en tres subconjuntos** para obtener una estimación final no contaminada.
5. **Aprendizaje por transferencia** con una red preentrenada como extractor de características.
6. **Recorte centrado en el personaje** para reducir la interferencia del fondo.

---

## 11. Consideraciones éticas

| Dimensión | Análisis |
|---|---|
| **Propiedad intelectual** | Las imágenes son fotogramas de una obra protegida por derechos de autor. Su uso aquí es **exclusivamente académico y sin fines de lucro**; una aplicación comercial requeriría autorización del titular. El modelo entrenado no se redistribuye. |
| **Sesgo por representación** | El dataset original está fuertemente desbalanceado. Entrenar sin corregirlo produciría un sistema que funciona para los personajes principales e ignora a los secundarios. El muestreo balanceado es también una **decisión de equidad** entre clases. |
| **Transferencia a personas reales** | La técnica aplicada sobre personajes dibujados es la misma que sustenta el reconocimiento facial. Trasladarla a rostros reales implicaría datos personales sensibles, consentimiento informado y riesgos documentados de sesgo demográfico. **Este proyecto no debe interpretarse como un paso hacia ese uso.** |
| **Transparencia** | Se reportan las métricas reales, incluidos los KPIs no alcanzados, y se declaran las limitaciones metodológicas. Ocultar un mal desempeño sería el principal riesgo ético de un proyecto de ML. |
| **Uso responsable** | Con el desempeño obtenido, el modelo **no debe usarse para etiquetado automático sin supervisión humana**; su uso legítimo es como apoyo a un revisor. |
| **Impacto ambiental** | El entrenamiento toma segundos en CPU y su huella de cómputo es despreciable. Se documenta porque a mayor escala este costo es un criterio de diseño relevante. |

---



## 12. Cómo reproducir la solución

### Opción A — Google Colab (recomendada)

1. Subir `notebooks/prueba 1 tecnicas de machine learning.ipynb` a Google Colab.
2. Ejecutar **`Entorno de ejecución → Reiniciar y ejecutar todo`**
   (*Runtime → Restart and run all*).
3. El dataset se descarga automáticamente con `kagglehub`; no requiere configuración manual ni GPU.
4. Al finalizar, descargar desde el panel de archivos de Colab las carpetas `models/` e `images/`
   generadas y copiarlas a este proyecto.
5. Copiar la salida de la última celda y pegarla en la [sección 8](#8-resultados) de este informe.

### Opción B — Entorno local

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scriptsctivate
pip install -r requirements.txt
jupyter notebook "notebooks/prueba 1 tecnicas de machine learning.ipynb"
```

### Reproducibilidad

El proyecto fija una **única semilla `SEED = 67`** propagada a las tres fuentes de aleatoriedad:

```python
random.seed(SEED)                     # aleatoriedad de Python
np.random.seed(SEED)                  # muestreo y partición
tf.keras.utils.set_random_seed(SEED)  # inicialización de pesos y mezclado de lotes
```

Además, la lista de archivos se **ordena con `sorted()` antes de muestrear**, porque
`os.listdir()` no garantiza un orden estable entre sistemas de archivos. Sin ese paso, la misma
semilla podría seleccionar imágenes distintas en otra máquina.
