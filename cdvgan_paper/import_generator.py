# Imports.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
from pathlib import Path

ruta_actual = Path(__file__).resolve()
ruta_proyecto = ruta_actual.parents[0]
if str(ruta_proyecto) not in sys.path:
    sys.path.insert(0, str(ruta_proyecto))

from .consistency_test import consistency_test
from .utils import plot_pca_3d

# ---------------------------------------------------------------------------
# Mismos parámetros que en main.py — deben coincidir EXACTAMENTE con los que
# se usaron para entrenar el generator.keras que vas a cargar (sobre todo
# noise_dim y num_classes/depth; si no coinciden, el generador va a fallar
# al recibir el input o va a generar basura silenciosamente).
# ---------------------------------------------------------------------------
noise_dim = 100
num_classes = 5
depth = num_classes

# ---------------------------------------------------------------------------
# Ruta al generador ya entrenado. Ajusta al que quieras evaluar:
#   - gan_exp_dir/<gan_choice>.keras   (el que guarda main.py al final)
#   - output_path/Generator.keras      (la copia dentro de GAN_outputs/<variant>/)
# ---------------------------------------------------------------------------
generator_path = f"{ruta_proyecto}/Generator.keras"  # <-- ajusta

generator = keras.models.load_model(generator_path)
print(f"Generador cargado desde: {generator_path}")
print("Input shape esperado por el generador:", generator.input_shape)

# ---------------------------------------------------------------------------
# Cargar el dataset REAL de test (el mismo que usa main.py para comparar) —
# esto NO es entrenamiento, solo carga de datos para la comparación.
# ---------------------------------------------------------------------------
data_orig = np.loadtxt(f"{ruta_proyecto}/data/corrected/ConditionalSignals_test.csv", delimiter=",")
class_array_orig = np.loadtxt(f"{ruta_proyecto}/data/corrected/ConditionalLabels_test.csv", delimiter=",")
class_array_orig = class_array_orig - 1

unique, counts_orig = np.unique(class_array_orig, return_counts=True)
print("Classes, counts (test real): ")
print(np.asarray((unique, counts_orig)).T)

# ---------------------------------------------------------------------------
# Armar las mismas clases one-hot y el mismo número de señales por clase que
# el dataset real de test, para que la comparación sea apples-to-apples.
# ---------------------------------------------------------------------------
num_signals_datasets = np.sum(counts_orig)

labels = np.concatenate([
    np.full(n, class_idx) for class_idx, n in enumerate(counts_orig)
])

vertex_classes_datasets = tf.one_hot(labels, depth, on_value=1.0, off_value=0.0, axis=-1)

# ---------------------------------------------------------------------------
# Generar directamente con el generador cargado — sin fit_GAN, sin GANMonitor,
# sin discriminadores. Esto es exactamente lo que hacía main.py al final,
# solo que usando el generador ya entrenado en vez de gan.generator recién
# salido de fit_GAN.
# ---------------------------------------------------------------------------
latent_vectors_vertex = tf.random.normal(shape=(num_signals_datasets, noise_dim))

generations_vertex = generator([latent_vectors_vertex, vertex_classes_datasets])
generations_vertex = generations_vertex.numpy()

print("Señales generadas:", generations_vertex.shape)

# ---------------------------------------------------------------------------
# Cross MMD / consistency test — igual que en main.py.
# ---------------------------------------------------------------------------
consistency_test(data_orig, class_array_orig, generations_vertex, labels)

plot_pca_3d(generations_vertex, labels)