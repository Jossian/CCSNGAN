import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from pathlib import Path
import time

ruta_proyecto = Path(__file__).resolve().parents[0]

# Deben coincidir EXACTAMENTE con los usados al entrenar
noise_dim = 100
num_classes = 5
depth = num_classes

generator_path = f"{ruta_proyecto}/Generator.keras"
generator = keras.models.load_model(generator_path)
print(f"Generador cargado desde: {generator_path}")
print("Input shape esperado:", generator.input_shape)

n_test = 10
fs_gan = 4096  # sampling rate al que generaste (según lo que confirmaste)

# Clases distribuidas ciclicamente entre las 5 clases para el set de prueba
labels = np.array([i % num_classes for i in range(n_test)])
vertex_classes = tf.one_hot(labels, depth, on_value=1.0, off_value=0.0, axis=-1)
latent_vectors = tf.random.normal(shape=(n_test, noise_dim))

t0 = time.time()
generations = generator([latent_vectors, vertex_classes])
generations = generations.numpy()
elapsed = time.time() - t0
print(f"Generadas {n_test} señales en {elapsed:.4f} s "
      f"({elapsed/n_test:.5f} s/señal — esto NO es el tiempo de PE, solo del generador)")
print("Shape crudo del generador:", generations.shape)

# El generador puede devolver (n, N) o (n, N, 1) según cómo esté armada la
# última capa — squeeze para dejarlo en (n_test, n_samples)
generations = np.squeeze(generations)
if generations.ndim == 1:
    generations = generations[np.newaxis, :]

n_samples = generations.shape[1]

# Vector de tiempo en ms, centrado en 0 (asumiendo bounce en el centro de la
# ventana, misma convención que Abylkairov_catalog.csv). Si tu GAN entrena
# con el bounce en otra posición del vector, ajusta el offset aquí.
t_ms = (np.arange(n_samples) - n_samples // 2) / fs_gan * 1000.0

rows = []
for s in range(n_test):
    for i in range(n_samples):
        rows.append({
            'sample_id': s,
            't(ms)': t_ms[i],
            'amplitude': generations[s, i],
            'class': labels[s],  # columna extra, load_gan_signals la ignora
        })

df = pd.DataFrame(rows)
out_path = ruta_proyecto / 'data' / 'gan_signals_test10.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Guardado en: {out_path}  ({n_test} señales x {n_samples} muestras)")