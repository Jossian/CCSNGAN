import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import backend as K
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, ks_2samp
from scipy.signal import correlate
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# Define the loss function for the discriminators,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_sig, fake_sig):
    """Calculate the Wasserstein Loss for discriminators.
    
    Parameters
    ---------

    real_sig: class: 'tf.Tensor'
            logits for real signals
            
    fake_sig: class: 'tf.Tensor'
            logits for fake signals
            
    Returns
    ---------
    
    Discriminator Wasserstein loss
    """
    real_loss = tf.reduce_mean(real_sig)
    fake_loss = tf.reduce_mean(fake_sig)
    return fake_loss - real_loss


# Define the loss function for the generator.
def generator_loss(fake_sig):
    """Calculate the Wasserstein Loss for generator.
    
    Parameters
    ---------     
    
    fake_sig: class: 'tf.Tensor'
            logits for fake signals
            
    Returns
    ---------
    
    Generator Wasserstein loss
    """
    return -tf.reduce_mean(fake_sig)

@tf.function
def calculate_derivative(x, y):
    """A tf function that calculates the derivative of one vector with respect to another.
    Used to calculate derivative signals of generated samples during training."""
    dydx = tf.experimental.numpy.diff(y)/tf.experimental.numpy.diff(x)
    return dydx

    
def plot_GAN_history(history, path, gan_variant):
    """A function to plot and save the WGAN or DVGAN losses to a certain path."""
    if gan_variant in ['cDVGAN', 'cDVGAN2', 'MCDVGANN']:
        plt.figure()
        plt.plot(history.history['d_loss'], 'b-', label = 'Discriminator 1')
        plt.plot(history.history['d2d_loss'], 'r-', label = 'Discriminator 2')
        plt.plot(history.history['g_loss'], 'g-', label = 'Generator 1')
        plt.plot(history.history['g_loss2d'], 'c-', label = 'Generator 2')
        plt.plot(history.history['g_loss_combined'], 'm-', label = 'Generator Combined')
        if gan_variant == 'cDVGAN2':
            plt.plot(history.history['d2d2_loss'], 'r-', label = 'Discriminator 3')
            plt.plot(history.history['g_loss2d2'], 'c-', label = 'Generator 3')
    else:
        plt.figure()
        plt.plot(history.history['d_loss'], 'b-', label = 'Discriminator')
        plt.plot(history.history['g_loss'], 'm-', label = 'Generator')
        
    plt.legend()
    plt.savefig(path+f'/{gan_variant}_loss_plot')
    plt.close()
    

def plot_examples(data, classes, path):
    """A function to plot and save 15 examples of training or generated data."""
    # Si classes es una lista/array de one-hot, convertir a enteros
    if isinstance(classes, (list, np.ndarray)) and np.ndim(classes) > 1:
        cls_numeric = np.argmax(classes, axis=1)
    else:
        cls_numeric = classes  # Ya es 1D, o es lista de enteros

    plt.figure(figsize=(12,12))
    for i in range(15):
        ax = plt.subplot(5, 3, i + 1)
        ax.plot(data[i])
        ax.set_ylim(-1, 1)

        # Establece el título con la clase correspondiente
        ax.set_title(f"Clase: {cls_numeric[i]}")
        ax.set_xlabel("Data points")
        ax.set_ylabel("Amplitud [A.U.]")

    plt.tight_layout()    
    plt.savefig(path)
    plt.show()
    plt.close()
def find_tbe(time_array, signal_median, n_crossings=2): # <--- CAMBIO CLAVE A n_crossings=2
    """
    Encuentra el tiempo del segundo cruce por cero (tbe) de la señal
    después del core bounce (t=0), siguiendo el ejemplo de la Figura 4.
    """
    
    # 1. Nos enfocamos solo en el tiempo post-bounce (t >= 0)
    post_bounce_mask = time_array >= 0
    t_post = time_array[post_bounce_mask]
    s_post = signal_median[post_bounce_mask]
    
    # 2. Encontrar los cruces por cero (cuando el signo cambia)
    zero_crossings_indices = np.where(np.diff(np.sign(s_post)))[0]
    
    if len(zero_crossings_indices) < n_crossings:
        # Si no se encuentra el N-ésimo cruce, devolvemos NaN.
        return np.nan 
    
    # 3. El índice del N-ésimo cruce por cero (ahora el segundo)
    tbe_index = zero_crossings_indices[n_crossings - 1]
    
    # 4. Devolver el tiempo (en segundos)
    return t_post[tbe_index]



def plot_examples_5(data, classes, path):
    """Plot and save one example per class (up to 5 classes) vertically."""
    # Convertir a etiquetas si es one-hot
    if isinstance(classes, (list, np.ndarray)) and np.ndim(classes) > 1:
        cls_numeric = np.argmax(classes, axis=1)
    else:
        cls_numeric = np.array(classes)

    # Obtener señales representativas: una por clase
    unique_classes = np.unique(cls_numeric)
    selected_signals = []
    selected_labels = []

    for c in unique_classes[:5]:  # máximo 5 clases
        idx = np.where(cls_numeric == c)[0]
        if len(idx) > 0:
            selected_signals.append(data[idx[0]])
            selected_labels.append(c)

    # Graficar verticalmente
    """
    plt.figure(figsize=(10, 10))
    for i, (signal, label) in enumerate(zip(selected_signals, selected_labels)):
        ax = plt.subplot(len(selected_signals), 1, i + 1)
        x = [j / 4096 for j in range(0, 256)]
        x = [value - (53 / 4096) for value in x]
        ax.plot(x, signal)
        ax.set_ylim(-1, 1)
        ax.set_title(f"Class: {int(label)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude [A.U.]")
    
    plt.suptitle("Original waveforms", fontsize=16)
    plt.tight_layout()
    plt.savefig(path+"/examples_org.png")
    plt.show()
    plt.close()"""
    plt.figure(figsize=(10, 6))

    # Definir un mapa de colores para las clases
    colors = plt.cm.tab10.colors  # hasta 10 colores distintos

    x = [j / 4096 - (53 / 4096) for j in range(256)]

    for signal, label in zip(selected_signals, selected_labels):
        color = colors[int(label) % len(colors)]  # asignar color por clase
        plt.plot(x, signal, label=f"Class {int(label)}", color=color, alpha=0.7)

    plt.ylim(-1, 1)
    plt.title("Original waveforms", fontsize=16)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude [A.U.]")

    # Para que no repita etiquetas en la leyenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(path + "/examples_org.png")
    plt.show()
    plt.close()
        
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_signal_distribution_by_class(signals, labels, time=None):
    """
    Genera subplots por clase mostrando la mediana, central 50% y central 95% de las señales,
    con ejes Y uniformes. Agrega líneas verticales para t=0 y t=tbounce.
    """

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    if time is None:
        time = [j / 4096 for j in range(0, 256)]
        time = [value - (53 / 4096) for value in time]
        time = np.array(time) 
    
    # 🌟 CORRECCIÓN DE LÍMITES Y UNIFORMES 🌟
    # Si bien usaste ax.set_ylim(-1, 1), es mejor calcularlo para robustez.
    # Pero mantendremos el -1, 1 para ser consistentes con tu código original.
    Y_LIMIT = 1.05 * np.max(np.abs(signals)) 
    # Usaremos el límite fijo de 1 si los datos lo permiten, o el calculado si se excede.
    if Y_LIMIT < 1.0: Y_LIMIT = 1.0
    
    # Crea el layout simple de N filas x 1 columna (compartiendo el eje X)
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 3.5 * n_classes), sharex=True)
    if n_classes == 1:
        axes = [axes]  # Asegura que axes sea iterable

    for idx, label in enumerate(unique_labels):
        ax = axes[idx]
        
        # Filtra las señales y los tiempos de rebote para la clase actual
        class_signals = signals[labels == label]
        #class_tbounce = tbounce[labels == label]
        #class_tbpostounce = tpbe[labels == label]

        # Evita errores si no hay datos
        if class_signals.size == 0:
             ax.set_title(f'Class: {label} (No data)')
             continue

        # Cálculo de percentiles (median, 50%, 95%)
        median = np.median(class_signals, axis=0)
        p25 = np.percentile(class_signals, 25, axis=0)
        p75 = np.percentile(class_signals, 75, axis=0)
        p2_5 = np.percentile(class_signals, 2.5, axis=0)
        p97_5 = np.percentile(class_signals, 97.5, axis=0)
        
        # Calcula el tiempo de rebote promedio para esta clase

        median_tbounce = find_tbe(time,median)

        # Plotting
        ax.fill_between(time, p2_5, p97_5, color='blue', alpha=0.2, label='Central 95%')
        ax.fill_between(time, p25, p75, color='blue', alpha=0.4, label='Central 50%')
        ax.plot(time, median, color='black', linewidth=1, label='Median of signals')
        
        # 🌟 AGREGAR LÍNEAS VERTICALES (core bounce) 🌟
        
        # 1. Línea vertical en t=0 (Inicio de la simulación o colapso)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, label=f'$t_b$')

        # 2. Línea vertical en t = 0 + tb (Rebote del núcleo)
        # Asumiendo que 'time' representa el tiempo después del inicio del colapso
        # y que tbounce (tb) ya está en unidades de tiempo consistentes con 'time'.
        ax.axvline(
            x=median_tbounce, 
            color='red', 
            linestyle='--', 
            linewidth=1.5, 
            label=f'$t_be$'
        )
        
        ax.set_ylabel('hD (cm)')
        ax.set_title(f'Class: {label}')
        
        # Aplica límites Y uniformes
        ax.set_ylim(-Y_LIMIT, Y_LIMIT)
        ax.grid(True)
        
        # Leyenda (incluyendo las líneas axvline)
        if idx == 0:
            # Combina todas las leyendas: median, 50%, 95%, t=0, t=tb
            handles, labels_list = ax.get_legend_handles_labels()
            # Ordena la leyenda para poner los rangos primero
            order = [0, 1, 2, 3, 4] # Orden típico: 95%, 50%, Median, t=0, t=tb
            ax.legend([handles[i] for i in order], [labels_list[i] for i in order], loc='upper right', fontsize='small')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def plot_pca_3d(X, labels, label_names=None, title="3D Projections: PCA, t-SNE, UMAP"):
    """
    Aplica PCA, t-SNE y UMAP a los datos X y grafica resultados en 3D con etiquetas.

    Parámetros:
    - X: ndarray o DataFrame de forma (n_samples, n_features)
    - labels: array-like de etiquetas de clase (n_samples,)
    - label_names: lista de nombres de las clases (opcional)
    - title: título general de la gráfica
    """
    fig = plt.figure(figsize=(18, 5))
    projections = []

    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_
    total_var_exp = np.sum(var_exp) * 100
    projections.append((X_pca, f"PCA\nVarianza total: {total_var_exp:.2f}%", 
                        [f"PC{i+1} ({v*100:.1f}%)" for i, v in enumerate(var_exp)]))

    # t-SNE
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    projections.append((X_tsne, "t-SNE", ["Dim 1", "Dim 2", "Dim 3"]))

    # UMAP
    reducer = umap.UMAP(n_components=3, random_state=42)
    X_umap = reducer.fit_transform(X)
    projections.append((X_umap, "UMAP", ["Dim 1", "Dim 2", "Dim 3"]))

    # Etiquetas legibles
    if label_names:
        label_map = dict(enumerate(label_names))
        labels_named = pd.Series(labels).map(label_map)
    else:
        labels_named = labels

    # Colores
    unique_labels = np.unique(labels_named)
    colors = sns.color_palette("husl", len(unique_labels))

    for i, (proj, subtitle, axis_labels) in enumerate(projections):
        df = pd.DataFrame(proj, columns=['X', 'Y', 'Z'])
        df['label'] = labels_named

        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        for j, label in enumerate(unique_labels):
            subset = df[df['label'] == label]
            ax.scatter(subset['X'], subset['Y'], subset['Z'],
                       label=label, s=15, alpha=0.7, color=colors[j])

        ax.set_title(subtitle)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        ax.legend(loc='upper right', fontsize='small')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def fit_GAN(GAN, data, batch_size, epochs, gan_variant = 'DVGAN', callback = False, noise_dim = 100, callback_path = 'DVGAN_monitor'):
    """A function to train the GAN model. We can choose to pass call backs to monitor the training after each epoch.
    
    Parameters
    ---------
    
    GAN: class: 'keras.Model'
            The GAN model to fit to data

    data: class:'numpy.ndarray' or 'tf.Tensor'
            A 1D array holding signals
            
    data_deriv: class:'numpy.ndarray' or 'tf.Tensor'
            A 1D array holding derivative signals
            
    batch_size: int
            The batch size for the model
        
    epochs: int
            The number of epochs to train the model
            
    gan_variant: str
            The variant of the GAN to be trained, either 'WGAN' or 'DVGAN'
          
    callback: bool
            Boolean controlling whether GAN monitor should be used
    
    callback_path: str
            The path where GAN monitor images are saved during training
            
    """

    if callback:
        # Instantiate the 'GANMonitor' for Keras callbacks.
        cbk = GANMonitor(num_img=1, latent_dim=noise_dim, gan_variant = gan_variant, callback_path=callback_path)
        isExist = os.path.exists(callback_path)
        if not isExist:
            os.makedirs(callback_path)
        
        history = GAN.fit(data, batch_size=batch_size, epochs=epochs, callbacks = [cbk])
    else:
        history = GAN.fit(data, batch_size=batch_size, epochs=epochs)
        
    return history
    
    
class GANMonitor(keras.callbacks.Callback):
    """GAN monitor used to plot GAN generated data after each epoch."""
    def __init__(self, num_img=1, latent_dim=100, gan_variant='WGAN', callback_path = 'WGAN_monitor'):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.gan_variant = gan_variant
        self.callback_path = callback_path

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
               
        indices = tf.experimental.numpy.random.randint(
                0,
                high=3,
                size=[self.num_img])
        depth = 5
        random_classes = tf.one_hot(indices, depth,
                  on_value=1.0, off_value=0.0,
                  axis=-1)
        generated_signals = self.model.generator([random_latent_vectors, random_classes])    
        

        for i in range(self.num_img):
            img = generated_signals[i].numpy()
            plt.figure()
            plt.plot(img)
            plt.savefig(self.callback_path+"/generated_img_{i}_{random_classes}_{epoch}.png".format(i=i, random_classes=random_classes, epoch=epoch),format="png")
            plt.close()
        
        # We will also save model components every 100 epochs
        if epoch%100==0:
            self.model.generator.save(self.callback_path+f'generator_{epoch}.keras')
            self.model.discriminator.save(self.callback_path+f'discriminator_{epoch}.keras')
            if self.gan_variant in ['DVGAN', 'DVGAN2', 'MCDVGANN']:
                self.model.deriv_discriminator.save(self.callback_path+f'deriv_discriminator_{epoch}.keras')
                if self.gan_variant == 'DVGAN2':
                    self.model.deriv2_discriminator.save(self.callback_path+f'deriv2_discriminator_{epoch}.keras')

            
            
def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def compare_signal_datasets(arr_original: np.ndarray, arr_augmented: np.ndarray, 
                            labels: np.ndarray, show_plots=True, max_pairs=1600):
    assert arr_original.shape[1] == arr_augmented.shape[1], "Las señales deben tener la misma longitud temporal"
    assert len(labels) == len(arr_original), "El número de etiquetas debe coincidir con el número de señales originales"

    print(f"🔎 Comparando {len(arr_original)} señales originales con {len(arr_augmented)} aumentadas...")

    def compute_stats(arr):
        return {
            "mean": np.mean(arr, axis=1),
            "std": np.std(arr, axis=1),
            "skewness": skew(arr, axis=1),
            "kurtosis": kurtosis(arr, axis=1)
        }

    stats_orig = compute_stats(arr_original)
    stats_aug = compute_stats(arr_augmented)

    print("\n📊 Estadísticas descriptivas:")
    for key in stats_orig:
        print(f"\n🔹 {key.upper()}")
        print(f" - Original:  mean={np.mean(stats_orig[key]):.4f}, std={np.std(stats_orig[key]):.4f}")
        print(f" - Aumentado: mean={np.mean(stats_aug[key]):.4f}, std={np.std(stats_aug[key]):.4f}")

    ks_pvalues = []
    for t in range(arr_original.shape[1]):
        stat, pval = ks_2samp(arr_original[:, t], arr_augmented[:, t])
        ks_pvalues.append(pval)

    avg_pval = np.mean(ks_pvalues)
    print(f"\n📈 Test de Kolmogorov-Smirnov promedio (por punto temporal): p = {avg_pval:.4f}")
    if avg_pval < 0.05:
        print("⚠️ Las distribuciones son significativamente diferentes en promedio (p < 0.05).")
    else:
        print("✅ Las distribuciones no son significativamente diferentes en promedio.")

    def normalized_xcorr(x, y):
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        return np.max(correlate(x, y, mode='full')) / len(x)

    n_pairs = min(max_pairs, len(arr_original), len(arr_augmented))
    idx_orig = random.sample(range(len(arr_original)), n_pairs)
    idx_aug = random.sample(range(len(arr_augmented)), n_pairs)

    xcorr_vals = [
        normalized_xcorr(arr_original[i], arr_augmented[j])
        for i, j in zip(idx_orig, idx_aug)
    ]
    print(f"\n🔗 Cross-correlation promedio (sobre {n_pairs} pares aleatorios): {np.mean(xcorr_vals):.4f}")

    if show_plots:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.ravel()

        axs[0].boxplot([stats_orig['mean'], stats_aug['mean']], labels=['Original', 'Aumentado'])
        axs[0].set_title("Media de señales")

        axs[1].boxplot([stats_orig['std'], stats_aug['std']], labels=['Original', 'Aumentado'])
        axs[1].set_title("Desviación estándar")

        axs[2].boxplot([stats_orig['skewness'], stats_aug['skewness']], labels=['Original', 'Aumentado'])
        axs[2].set_title("Asimetría (Skewness)")

        axs[3].hist(xcorr_vals, bins=20, color='skyblue', edgecolor='k')
        axs[3].set_title("Distribución de Cross-Correlation")

        plt.tight_layout()
        plt.show()

          # --- NUEVA SECCIÓN: graficar 3 señales originales por clase en una sola figura 5x3 ---
        print("\n📉 Mostrando 3 señales originales por clase (en una sola figura 5x3)...")
        unique_classes = np.unique(labels)
        fig, axs = plt.subplots(5, 1, sharex=True, sharey=True)
        axs = axs.ravel()
        
        plot_idx = 0
        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
            selected_indices = np.random.choice(class_indices, size=min(1, len(class_indices)), replace=False)
            
            for i, idx in enumerate(selected_indices):
                ax = axs[plot_idx]
                x = [j / 4096 for j in range(0, 256)]
                x = [value - (53 / 4096) for value in x]

                ax.plot(x,arr_original[idx], color='blue')
                

                ax.set_title(f"Class {class_label}")
                ax.set_ylim(-1, 1)
                ax.set_ylabel("Amplitude [A.U.]")
                ax.grid(True)
                plot_idx += 1
                if plot_idx >= 15:
                    break
            if plot_idx >= 15:
                break
        
        for ax in axs:
            ax.set_xlabel("Time ()")

        plt.tight_layout()
        plt.show()
        #plt.savefig(+f'/examples_plot')

        plot_pca_3d(arr_original,labels)