# Imports.
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import sys
from pathlib import Path
# Obtén la ruta del archivo actual
ruta_actual = Path(__file__).resolve()
ruta_proyecto = ruta_actual.parents[0]  # .parents[0] es el archivo mismo, .parents[1] es el padre, etc.
if str(ruta_proyecto) not in sys.path:
    sys.path.insert(0, str(ruta_proyecto))
import time
import pandas as pd

from .gan_models import choose_gan
from .utils import discriminator_loss, generator_loss, calculate_derivative, plot_GAN_history, plot_examples, plot_examples_5,plot_examples_complete, GANMonitor, fit_GAN, compare_signal_datasets, plot_pca_3d, plot_signal_distribution_by_class
from .consistency_test import consistency_test
import pickle
import json
import os

#print(tf.config.list_physical_devices('GPU'))

# Set parameters.
sample_rate = 1024
noise_dim = 100
num_classes = 5
depth = num_classes
#-----------------------------------------------------------------------------------
# Set directories for storing outputs

output_dir = f'{ruta_proyecto}/GAN_outputs/'
monitor_dir =f'{ruta_proyecto}/Monitor/'
gan_exp_dir =f'{ruta_proyecto}/GAN_experiments/Trained_CGANs/'
"""
with open(f'{ruta_proyecto}/data/data.pkl', 'rb') as f:
  data = pickle.load(f)
  
with open(f'{ruta_proyecto}/data/data_deriv.pkl', 'rb') as f:
  data_deriv = pickle.load(f)

with open(f'{ruta_proyecto}/data/data_deriv2.pkl', 'rb') as f:
  data_deriv2 = pickle.load(f)
  
with open(f'{ruta_proyecto}/data/class_array.pkl', 'rb') as f:
  class_array = pickle.load(f)
"""
#------------------------------------------------------------------------------------
#data = np.loadtxt(f'{ruta_proyecto}/data/ConditionalSignals_aug_beta.csv', delimiter=',')
#class_array = np.loadtxt(f'{ruta_proyecto}/data/ConditionalLabels_aug_beta.csv', delimiter=',')

data = np.loadtxt(f'{ruta_proyecto}/data/corrected/ConditionalSignals_aug_train.csv', delimiter=',')
class_array = np.loadtxt(f'{ruta_proyecto}/data/corrected/ConditionalLabels_aug_train.csv', delimiter=',')


data_signals=data
class_array = class_array - 1
unique,counts=np.unique(class_array , return_counts=True)
#print("Classes, counts  originales del dataset: ")
#print(np.asarray((unique, counts)).T)



data_orig=np.loadtxt(f'{ruta_proyecto}/data/corrected/ConditionalSignals_test.csv', delimiter=',')
class_array_orig = np.loadtxt(f'{ruta_proyecto}/data/corrected/ConditionalLabels_test.csv', delimiter=',')
class_array_orig = class_array_orig - 1

unique,counts_orig=np.unique(class_array_orig, return_counts=True)
#print("Classes, counts  originales del dataset: ")
#print(np.asarray((unique, counts)).T)
#parameters=np.loadtxt(f'{ruta_proyecto}/data/labels.csv', delimiter=',',skiprows=1,usecols=(4,5,6))

# Column names corresponding to indices 4, 5, and 6 in your labels.csv
# Assuming the order is: beta1_IC_b (4), tbounce_s (5), tbe_calculated_s (6)
COLS_TO_LOAD = ["beta1_IC_b", "tbounce_s", "t_postbounce"]

# 1. Load the CSV into a temporary Pandas DataFrame, only reading the necessary columns.
labels_df_temp = pd.read_csv(f'{ruta_proyecto}/data/labels.csv', usecols=COLS_TO_LOAD)

# 2. Convert the DataFrame (including the NaNs converted by Pandas) into a NumPy array.
# The NaNs will now be stored as valid numpy.nan float values.
parameters = labels_df_temp.values 


#signals_richers=np.loadtxt(f'{ruta_proyecto}/data/signals_preprocessed.csv', delimiter=',',skiprows=1)
#time_interp=np.loadtxt(f'{ruta_proyecto}/data/time_interpolated.csv', delimiter=',',skiprows=1)

##converting labels of richers dataset to plot examples
# Extraer la columna de beta
#beta = parameters[:, 0]
#beta = labels_df_temp['beta1_IC_b'].values
# --- 2. Definición de condiciones y valores de etiquetas ---

# 2a. Definir las condiciones booleanas de las clases (en el orden lógico)
# Nota: np.select evalúa las condiciones en orden.
"""
conditions = [
    # Clase 1: beta <= 0.02
    (beta <= 0.02),
    
    # Clase 2: 0.02 < beta <= 0.04
    (beta > 0.02) & (beta <= 0.04),
    
    # Clase 3: 0.04 < beta <= 0.06
    (beta > 0.04) & (beta <= 0.06),
    
    # Clase 4: 0.06 < beta <= 0.17
    (beta > 0.06) & (beta <= 0.17),
    
    # Clase 5: beta > 0.17
    (beta > 0.17) 
]

# 2b. Definir los valores que se asignarán a cada condición
choices = [
    1,  # Valor para Clase 1
    2,  # Valor para Clase 2
    3,  # Valor para Clase 3
    4,  # Valor para Clase 4
    5   # Valor para Clase 5
]

# --- 3. Aplicar np.select para crear el nuevo array de etiquetas ---

# El argumento 'default' (aquí 0) se usa para cualquier valor que no cumpla ninguna condición.
# Aunque el conjunto de condiciones cubre todos los números reales, es buena práctica incluirlo.
new_labels = np.select(conditions, choices, default=0)
print("unique labels richers: ")
print(np.unique(new_labels))
# --- 4. Resultado ---
print("Valores de Beta:")
print(beta)
print("\nNuevo Array de Etiquetas (new_labels):")
print(new_labels)
########################################################3"""




#tbounce=tbounce['tbounce_s']
plot_signal_distribution_by_class(data_orig,class_array_orig,f'{ruta_proyecto}/GAN_outputs/')
metrics=compare_signal_datasets(data_orig,data,class_array_orig)
plot_examples_5(data_orig, class_array_orig, monitor_dir)


#class_array = class_array.astype(np.int32)
class_array_one_hot = tf.one_hot(class_array, depth)


zero_rows = tf.reduce_all(tf.equal(class_array_one_hot, 0), axis=1)
#print("Número de filas con [0, 0, 0]:", tf.reduce_sum(tf.cast(zero_rows, tf.int32)).numpy())

#print("class_array shape: ", class_array.shape)

# Paso 1: calcular las derivadas si no las tienes
#tf.cast(np.array(range(256)), tf.float32), fake_signals

#x = np.arange(256)
data_deriv = calculate_derivative(tf.cast(np.array(range(256)), tf.float64), data)

#x = np.arange(256)
data_deriv2 = calculate_derivative(tf.cast(np.array(range(255)), tf.float64), data_deriv)

# Paso 2: empacar en el formato esperado
#data = ([signals, derivatives, labels])


#------------------------------------------------------------------------------------
gan_choice_dict = {1:'cWGAN',
                  2:'cDVGAN',
                  3:'cDVGAN2',
                  4:'MCGANN',
                  5:'MCDVGANN'}
gan_choice_int = int(input('Please input the GAN variant to train (1: cWGAN, 2: cDVGAN, 3: cDVGAN2, 4: MCGANN, 5: MCDVGANN): '))
gan_choice = gan_choice_dict[gan_choice_int]

# Change the below to 'WGAN' to train a vanilla model
signal_length = data.shape[-1]
deriv_signal_length = data_deriv.shape[-1]
deriv2_signal_length = data_deriv2.shape[-1]
#print("deriv_signal_length: ",deriv_signal_length)
# Create and compile GAN model
gan = choose_gan(gan_choice, signal_length, deriv_signal_length, deriv2_signal_length, num_classes, noise_dim)

# Set batch size and number of epochs for training.
BATCH_SIZE = 16
epochs = 500

# Change to False if you don't want the GAN monitor to plot generated signals after each epoch.
callback = True
# Callback path to save GAN monitor signals.
callback_path = monitor_dir+gan_choice+'/'

if gan_choice in ['cDVGAN', 'MCDVGANN']:
    data = [data, data_deriv, class_array_one_hot]
elif gan_choice == 'cDVGAN2':
    data = [data, data_deriv, data_deriv2, class_array_one_hot]
else:
    data = [data, class_array_one_hot]
    
# Start training the model, saving the histroy information.
start = time.time()
history = fit_GAN(gan, data, batch_size=BATCH_SIZE, epochs=epochs, gan_variant = gan_choice, callback = callback, noise_dim = noise_dim, callback_path=callback_path)
end = time.time()
print(f"Training time: {end - start:.6f} seconds")

# Output path where loss plots, generated examples and trained generators are stored.
output_path = output_dir+gan_choice
isExist = os.path.exists(output_path)
if not isExist:
    os.makedirs(output_path)

isExist = os.path.exists(gan_exp_dir)
if not isExist:
    os.makedirs(gan_exp_dir)
    
# Plot training history.
plot_GAN_history(history, output_path, gan_choice)

# Save history as json.
# Get the dictionary containing each metric and the loss for each epoch.
history_dict = history.history
# Dump it
json.dump(history_dict, open(output_path+'/history.json', 'w'))

# Save the generator for experiments.
gan.generator.save(f'{gan_exp_dir}/{gan_choice.lower()}.keras')

# Save all components.
print("generator saved on: ", output_path)
gan.generator.save(output_path+'/Generator.keras')
gan.discriminator.save(output_path+'/Discriminator.keras')
if gan_choice in ['DVGAN', 'DVGAN2', 'MCDVGANN']:
    gan.deriv_discriminator.save(output_path+'/Deriv_Discriminator.keras')
    if gan_choice == 'DVGAN2':
        gan.deriv2_discriminator.save(output_path+'/Deriv2_Discriminator.keras')

        


# Generate signals. We will sample the class space in three different ways; vertex, simplex and uniform sampling
num_signals = 50
'''indices = tf.experimental.numpy.random.randint(
        0,
        high=num_classes,
        size=[num_signals])'''


indices = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4])
vertex_classes = tf.one_hot(indices, depth,
          on_value=1.0, off_value=0.0,
          axis=-1)

random_ints = np.random.randint(0, 100, size=(num_signals,num_classes))

simplex_classes = random_ints/np.sum(random_ints, axis=1).reshape(num_signals,1)
simplex_classes = tf.convert_to_tensor(simplex_classes, dtype=tf.float32)

uniform_classes = np.random.uniform(low=0.0, high=1.0, size=(num_signals,num_classes))
uniform_classes = tf.convert_to_tensor(uniform_classes, dtype=tf.float32)


latent_vectors_vertex = tf.random.normal(shape=(num_signals, noise_dim))
latent_vectors_simplex = tf.random.normal(shape=(num_signals, noise_dim))
latent_vectors_uniform = tf.random.normal(shape=(num_signals, noise_dim))

start = time.time()
generations_vertex = gan.generator([latent_vectors_vertex, vertex_classes])
end = time.time()
print(f"Tiempo de generación para 15 señales: {end - start:.6f} segundos")

generations_vertex = generations_vertex.numpy()

#generations_simplex = gan.generator([latent_vectors_vertex, simplex_classes])
#generations_simplex = generations_simplex.numpy()

generations_uniform = gan.generator([latent_vectors_vertex, uniform_classes])
generations_uniform = generations_uniform.numpy()

vertex_classes = vertex_classes.numpy()
#simplex_classes = simplex_classes.numpy()
#uniform_classes = uniform_classes.numpy()

#print("vertex classes: ", vertex_classes)
# Plot some examples of generated data using different sampling methods.
#plot_examples_5(generations_vertex, vertex_classes, output_path)
plot_examples_complete(generations_vertex, vertex_classes, output_path)

#plot_examples(generations_vertex, vertex_classes, output_path+'/Vertex_examples')
#plot_examples(generations_simplex, simplex_classes, output_path+'/Simplex_examples')
#plot_examples(generations_uniform, uniform_classes, output_path+'/Uniform_examples')


print('Training Completed!')

print("Generating datasets to consistency test ...")

#samples_per_class = 1000
# Crear etiquetas como números enteros: 0, 1, 2 repetidos 1000 veces cada uno
#labels = np.repeat(np.arange(num_classes), samples_per_class)
# Convertir a one-hot encoding
#vertex_classes_datasets = np.eye(num_classes)[labels]

#print(vertex_classes_datasets.shape)  # (3000, 3)
#num_signals_datasets=3000




# Especificar el número de muestras por clase (por ejemplo: clase 0: 800, clase 1: 1200, clase 2: 500)
#samples_per_class = [800, 1200, 500]
num_signals_datasets = np.sum(counts_orig)

# Crear etiquetas de clase
labels = np.concatenate([
    np.full(n, class_idx) for class_idx, n in enumerate(counts_orig)
])

unique,counts=np.unique(labels, return_counts=True)
print("Classes, counts: ")
print(np.asarray((unique, counts)).T)
print("counts: ", counts)

# Convertir a one-hot encoding
#vertex_classes_datasets = np.eye(num_signals_datasets)[labels]

vertex_classes_datasets = tf.one_hot(labels, depth,
          on_value=1.0, off_value=0.0,
          axis=-1)
zero_rows = tf.reduce_all(tf.equal(vertex_classes_datasets, 0), axis=1)
#print("Número de filas con [0, 0, 0]:", tf.reduce_sum(tf.cast(zero_rows, tf.int32)).numpy())

latent_vectors_vertex = tf.random.normal(shape=(num_signals_datasets, noise_dim))

generations_vertex = gan.generator([latent_vectors_vertex, vertex_classes_datasets])

generations_vertex = generations_vertex.numpy()


#print("lenght generations: ", generations_vertex.shape)
consistency_test(data_orig, class_array_orig, generations_vertex, labels)

plot_pca_3d(generations_vertex, labels)