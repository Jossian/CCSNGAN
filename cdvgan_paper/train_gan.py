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


from .gan_models import choose_gan
from .utils import discriminator_loss, generator_loss, calculate_derivative, plot_GAN_history, plot_examples, GANMonitor, fit_GAN

import pickle
import json
import os

print(tf.config.list_physical_devices('GPU'))

# Set parameters.
sample_rate = 1024
noise_dim = 100
num_classes = 3
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
data = np.loadtxt(f'{ruta_proyecto}/data/ConditionalSignals.csv', delimiter=',')
class_array = np.loadtxt(f'{ruta_proyecto}/data/ConditionalLabels.csv', delimiter=',')


mask = class_array != 0
data = data[mask]
class_array = class_array[mask]

class_array = tf.one_hot(class_array, depth=3)
#print("class_array shape: ", class_array.shape)

# Paso 1: calcular las derivadas si no las tienes
def calculate_derivative(x, y):
    return np.gradient(y, x, axis=1)

x = np.arange(256)
data_deriv = calculate_derivative(x, data)

x = np.arange(256)
data_deriv2 = calculate_derivative(x, data_deriv)

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
print("deriv_signal_length: ",deriv_signal_length)
# Create and compile GAN model
gan = choose_gan(gan_choice, signal_length, deriv_signal_length, deriv2_signal_length, num_classes, noise_dim)

# Set batch size and number of epochs for training.
BATCH_SIZE = 512
epochs = 5

# Change to False if you don't want the GAN monitor to plot generated signals after each epoch.
callback = True
# Callback path to save GAN monitor signals.
callback_path = monitor_dir+gan_choice+'/'

if gan_choice in ['cDVGAN', 'MCDVGANN']:
    data = [data, data_deriv, class_array]
elif gan_choice == 'cDVGAN2':
    data = [data, data_deriv, data_deriv2, class_array]
else:
    data = [data, class_array]
    
# Start training the model, saving the histroy information.
history = fit_GAN(gan, data, batch_size=BATCH_SIZE, epochs=epochs, gan_variant = gan_choice, callback = callback, noise_dim = noise_dim, callback_path=callback_path)

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
gan.generator.save(output_path+'/Generator.keras')
gan.discriminator.save(output_path+'/Discriminator.keras')
if gan_choice in ['DVGAN', 'DVGAN2', 'MCDVGANN']:
    gan.deriv_discriminator.save(output_path+'/Deriv_Discriminator.keras')
    if gan_choice == 'DVGAN2':
        gan.deriv2_discriminator.save(output_path+'/Deriv2_Discriminator.keras')

        


# Generate signals. We will sample the class space in three different ways; vertex, simplex and uniform sampling
num_signals = 10
indices = tf.experimental.numpy.random.randint(
        0,
        high=num_classes,
        size=[num_signals])
depth = num_classes

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

generations_vertex = gan.generator([latent_vectors_vertex, vertex_classes])
generations_vertex = generations_vertex.numpy()

generations_simplex = gan.generator([latent_vectors_vertex, simplex_classes])
generations_simplex = generations_simplex.numpy()

generations_uniform = gan.generator([latent_vectors_vertex, uniform_classes])
generations_uniform = generations_uniform.numpy()

vertex_classes = vertex_classes.numpy()
simplex_classes = simplex_classes.numpy()
uniform_classes = uniform_classes.numpy()

# Plot some examples of generated data using different sampling methods.
plot_examples(generations_vertex, vertex_classes, output_path+'/Vertex_examples')
plot_examples(generations_simplex, simplex_classes, output_path+'/Simplex_examples')
plot_examples(generations_uniform, uniform_classes, output_path+'/Uniform_examples')


print('Training Completed!')
