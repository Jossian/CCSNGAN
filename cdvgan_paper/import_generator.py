"""
Carga un generador ya entrenado (.keras) y corre únicamente el consistency
test contra el set de test original, generando el mismo número de señales
por clase que hay en ConditionalLabels_test.csv (igual que al final del
script de entrenamiento).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

ruta_actual = Path(__file__).resolve()
ruta_proyecto = ruta_actual.parents[0]

from .consistency_test import consistency_test
from .utils import plot_pca_3d

# ------------------------------------------------------------------
# Parámetros — deben coincidir EXACTAMENTE con los usados al entrenar
# ------------------------------------------------------------------
noise_dim = 100
num_classes = 5
depth = num_classes

# ------------------------------------------------------------------
# Elegir qué generador cargar
# ------------------------------------------------------------------
gan_choice_dict = {1: 'cWGAN',
                    2: 'cDVGAN',
                    3: 'cDVGAN2',
                    4: 'MCGANN',
                    5: 'MCDVGANN'}
gan_choice_int = int(input(
    'Please input the GAN variant whose generator to load '
    '(1: cWGAN, 2: cDVGAN, 3: cDVGAN2, 4: MCGANN, 5: MCDVGANN): '))
gan_choice = gan_choice_dict[gan_choice_int]

output_dir = f'{ruta_proyecto}/GAN_outputs/'
output_path = output_dir + gan_choice

generator_path = output_path + '/Generator.keras'
generator = keras.models.load_model("Generator.keras")
print(f"Generador cargado desde: {"Generator.keras"}")
print("Input shape esperado:", generator.input_shape)

# ------------------------------------------------------------------
# Cargar el set de test original (referencia real para el consistency test)
# ------------------------------------------------------------------
data_orig = np.loadtxt(f'{ruta_proyecto}/data/corrected/ConditionalSignals_test.csv', delimiter=',')
class_array_orig = np.loadtxt(f'{ruta_proyecto}/data/corrected/ConditionalLabels_test.csv', delimiter=',')
class_array_orig = class_array_orig - 1

unique, counts_orig = np.unique(class_array_orig, return_counts=True)
print("Classes, counts originales del dataset de test: ")
print(np.asarray((unique, counts_orig)).T)

# ------------------------------------------------------------------
# Generar el mismo número de señales por clase que en el set de test
# ------------------------------------------------------------------
num_signals_datasets = np.sum(counts_orig)

labels = np.concatenate([
    np.full(n, class_idx) for class_idx, n in enumerate(counts_orig)
])

unique, counts = np.unique(labels, return_counts=True)
print("Classes, counts (generación): ")
print(np.asarray((unique, counts)).T)

vertex_classes_datasets = tf.one_hot(labels, depth,
                                      on_value=1.0, off_value=0.0,
                                      axis=-1)

latent_vectors_vertex = tf.random.normal(shape=(num_signals_datasets, noise_dim))

generations_vertex = generator([latent_vectors_vertex, vertex_classes_datasets])
generations_vertex = generations_vertex.numpy()

print("Shape de las señales generadas:", generations_vertex.shape)

# ------------------------------------------------------------------
# Consistency test
# ------------------------------------------------------------------
consistency_test(data_orig, class_array_orig, generations_vertex, labels)

plot_pca_3d(generations_vertex, labels)

print("Consistency test completado.")