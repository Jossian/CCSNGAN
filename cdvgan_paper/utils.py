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

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(path)
    plt.show()
    plt.close()


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
    plt.figure(figsize=(10, 10))
    for i, (signal, label) in enumerate(zip(selected_signals, selected_labels)):
        ax = plt.subplot(len(selected_signals), 1, i + 1)
        ax.plot(signal)
        ax.set_ylim(-1, 1)
        ax.set_title(f"Clase: {label}")
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Amplitud [A.U.]")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()
    
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        fig, axs = plt.subplots(5, 3, figsize=(15, 10), sharex=True, sharey=True)
        axs = axs.ravel()
        
        plot_idx = 0
        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
            selected_indices = np.random.choice(class_indices, size=min(3, len(class_indices)), replace=False)
            
            for i, idx in enumerate(selected_indices):
                ax = axs[plot_idx]
                ax.plot(arr_original[idx], color='blue')
                ax.set_title(f"Clase {class_label}")
                ax.set_ylim(-1, 1)
                ax.set_ylabel("Amplitud")
                ax.grid(True)
                plot_idx += 1
                if plot_idx >= 15:
                    break
            if plot_idx >= 15:
                break
        
        for ax in axs:
            ax.set_xlabel("Tiempo")

        plt.tight_layout()
        plt.show()
        plot_pca_3d(arr_original,labels)