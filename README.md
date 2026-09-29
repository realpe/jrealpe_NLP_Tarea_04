# Mini-proyecto 4 — Generación controlada de reseñas en español con GPT-2

**Autores:** José Luis Realpe M., Alejandra Forero, Santiago Aristizabal, Sandra Orozco
**Curso:** Procesamiento de Lenguaje Natural — Maestría en IA Aplicada, Universidad Icesi
**Basado en:** notebook guía `1-text-generation.ipynb` (Sesión 4), Radford et al. (2018) y Keskar et al. (2019), *CTRL*

## Problema

**Generar reseñas de producto en español con la calificación que se pida** (`mteb/amazon_reviews_multi`, config `es`) usando
`mrm8488/spanish-gpt2` (124M parámetros), y medir la calidad del texto generado con métricas, ya que el notebook guía evalúa
leyendo una muestra. La calificación de 1 a 5 estrellas pasa de etiqueta a **instrucción**: cada reseña de entrenamiento
lleva al inicio la línea `Calificación: N estrellas` (código de control) y, al generar, esa línea pide la calificación.
Cuatro experimentos responden a cuatro preguntas:

1. ¿El fine-tuning del guía produce un generador que sabe terminar un texto?
2. ¿Cómo cambia el texto según la estrategia de decodificación?
3. ¿Puede la calificación controlar el sentimiento del texto generado?
4. ¿Se puede distinguir el texto generado del real?

## Contenido

| Archivo | Descripción |
|---|---|
| `mini_proyecto_generacion_texto.ipynb` | Notebook para Google Colab con GPU: EDA orientado al generador (longitud en tokens, vocabulario polar, fórmulas repetidas), diagnóstico del collator, juez automático (TF-IDF + regresión logística), los cuatro experimentos, demo interactiva y conclusiones |
| `README.md` | Este archivo |

## Resultados principales (corrida completa, A100)

**Experimento 1 — receta de fine-tuning** (30 000 reseñas, 4 épocas):

| Modelo | Perplejidad | P(EOS) al final natural | Textos que terminan solos |
|---|---|---|---|
| Base sin ajustar | 282.2 | 1.8e-7 | 0.00 |
| Receta del guía (`pad = eos`, sin EOS) | 29.77 | 4.9e-9 | 0.00 |
| Receta corregida (relleno propio, EOS explícito) | 30.57 | 0.310 | 0.92 |

**Experimento 2 — decodificación** (distinct-2 / repetición de trigramas / palabras):
greedy 0.268 / 0.206 / 23.5; beam 4 0.411 / 0.127 / 13.0; temperatura 1.0 0.857 / 0.006 / 27.7; **top-p 0.9 0.818 / 0.006 / 30.0**;
top-p 0.9 + penalización 1.2 0.853 / 0.0007 / 30.0, frente a las reseñas reales 0.811 / 0.0125 / 31.3.

**Experimento 3 — control por calificación** (top-p 0.9; proporción de textos que el juez clasifica en la clase pedida):

| Calificación | 1 ★ | 2 ★ | 3 ★ | 4 ★ | 5 ★ | Global |
|---|---|---|---|---|---|---|
| Texto generado | 0.85 | 0.69 | 0.34 | 0.69 | 0.86 | 0.686 |
| Reseñas reales (referencia del juez) | 0.916 | 0.667 | 0.503 | 0.712 | 0.905 | 0.741 |

Correlación de Spearman entre la calificación pedida y la estimada por el juez: 0.739.

**Experimento 4 — detección** (AUC de un detector TF-IDF + regresión logística): greedy **0.964** (0.836 solo con la longitud),
temperatura 1.0 **0.540**, top-p + penalización **0.546**.

Lecturas clave: el EOS explícito y el relleno propio hacen que 92 % de los textos terminen solos, a un costo de +0.8 de
perplejidad. Greedy y beam search repiten o acortan el texto; top-p 0.9 queda más cerca de la distribución real, y con penalización de
repetición la repetición y la copia del corpus caen por debajo de las reales. Un prefijo de texto dirige el sentimiento con claridad en 1 y 5 estrellas y con dificultad en 3. El texto
de greedy se detecta con facilidad, mientras que el muestreo top-p con penalización no deja marcas que un detector léxico
sencillo pueda usar.

## Lectura crítica del notebook guía

| Notebook guía | Qué hicimos |
|---|---|
| `pad_token = eos_token`: el collator ignora todos los EOS | Diagnóstico de las etiquetas y relleno propio |
| El tokenizer no agrega EOS: el modelo no aprende a terminar | EOS explícito al final de cada reseña completa |
| 2 419 chistes, lote de 2, 10 épocas y sobreajuste después de la época 3 | 30 000 reseñas, 4 épocas, early stopping y checkpoint por validación |
| Una muestra por modelo como evaluación | Perplejidad, entropía, diversidad, repetición, copia de n-gramas y juez automático sobre cientos de textos con semilla fija |
| Sin exploración de top-p ni penalización de repetición | Siete configuraciones de decodificación comparadas |
| Metadatos (categoría, humor) sin usar | Calificación como código de control |
| Sin análisis del riesgo del texto generado | Detector de texto sintético |

## Cómo ejecutarlo

**Google Colab (recomendado):** abrir el notebook, seleccionar *Entorno de ejecución → Cambiar tipo de entorno → GPU* y
ejecutar todas las celdas. En A100 los tres fine-tunings suman unos 12 minutos y la corrida completa cerca de media hora; en L4
se estiman 40 a 60 minutos. Con `QUICK_MODE = True` la corrida baja a unos 10 minutos con menos datos (las cifras no son
comparables). Los modelos ajustados se guardan en Google Drive (unos 0.5 GB por modelo) y se reutilizan si Colab se
desconecta. Si el Hub responde con error 429, guardar un token de lectura de HuggingFace como secreto `HF_TOKEN`.

**Local:**

```bash
pip install torch datasets "transformers>=4.46" accelerate scikit-learn pandas matplotlib seaborn ipywidgets huggingface_hub
jupyter notebook
```

Sin GPU el notebook activa `QUICK_MODE` por su cuenta.

## Estructura del notebook

1. Planteamiento: problema, preguntas, hipótesis y diseño
2. Entorno
3. Datos
4. Análisis exploratorio: reseñas por calificación, longitud en tokens y `MAX_LEN`, vocabulario polar, fórmulas repetidas
5. El modelo y el detalle del collator en el notebook guía
6. Utilidades, métricas y juez automático
7. Experimento 1: receta del guía frente a receta corregida
8. Experimento 2: estrategias de decodificación
9. Experimento 3: generación controlada por calificación
10. Experimento 4: detección de texto generado
11. Demo interactiva
12. Conclusiones

## Referencias

- Radford, A. et al. (2018). *Improving Language Understanding by Generative Pre-Training*. OpenAI.
- Keskar, N. et al. (2019). *CTRL: A Conditional Transformer Language Model for Controllable Generation*. arXiv:1909.05858.
- Holtzman, A. et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR.
- Fan, A. et al. (2018). *Hierarchical Neural Story Generation*. ACL.
- Li, J. et al. (2016). *A Diversity-Promoting Objective Function for Neural Conversation Models*. NAACL.
- Keung, P. et al. (2020). *The Multilingual Amazon Reviews Corpus*. EMNLP.
- Tunstall, von Werra & Wolf (2022). *Natural Language Processing with Transformers*. O'Reilly.
- Notebooks guía del curso: https://github.com/Ohtar10/icesi-nlp
