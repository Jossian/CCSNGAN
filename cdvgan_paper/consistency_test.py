import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import correlate
from matplotlib.gridspec import GridSpec



# Simulación de poblaciones (real vs artificial) para cada clase
def simulate_blips(n, dim=50):
    return np.random.normal(0, 1, size=(n, dim))

# Métrica de similitud: Wasserstein (W1), match function (Mf), cross-covariance (k)
def match_function(x, y):
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return np.dot(x, y)

# Optimized normalized cross-covariance using direct formula
def normalized_cross_covariance_fast(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return numerator / denominator if denominator != 0 else 0

# Replace slow version
def compute_similarity_fast(blip, population, metric='wasserstein'):
    sims = []
    for sample in population:
        if metric == 'wasserstein':
            sims.append(wasserstein_distance(blip, sample))
        elif metric == 'match':
            sims.append(match_function(blip, sample))
        elif metric == 'crosscov':
            sims.append(normalized_cross_covariance_fast(blip, sample))
    return np.mean(sims), np.std(sims) / np.sqrt(len(sims))

# Redefinir función de ploteo
def plot_similarity_metric(classes, results, metric_name, xlabel, ylabel, invert=False):
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    colors = {0: 'royalblue', 1: 'darkorange', 2: 'forestgreen'}

    for cls in classes:
        cls_mask = np.array(results[metric_name]['label']) == cls
        x = np.array(results[metric_name]['x'])[cls_mask]
        y = np.array(results[metric_name]['y'])[cls_mask]

        color = colors.get(cls, 'gray')
        ax_main.scatter(x, y, label=cls, alpha=0.5, s=10, color=color)

        # Ajuste lineal
        slope, intercept, *_ = linregress(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        ax_main.plot(x_line, slope * x_line + intercept, color=color,
                     label=fr"{cls}: $y = {slope:.2f}x + {intercept:.2f}$")

        # Histograma superior e izquierdo
        ax_histx.hist(x, bins=50, color=color, alpha=0.6)
        ax_histy.hist(y, bins=50, color=color, orientation='horizontal', alpha=0.6)

        # Intervalo de confianza ±6σ
        x_mean, x_std = np.mean(x), np.std(x)
        y_mean, y_std = np.mean(y), np.std(y)
        for delta in [-6, 6]:
            ax_main.axvline(x_mean + delta * x_std, linestyle='--', color=color, alpha=0.4)
            ax_main.axhline(y_mean + delta * y_std, linestyle='--', color=color, alpha=0.4)

    # Ejes y estilo
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(f"Similarity: {metric_name.capitalize()}")
    if invert:
        ax_main.invert_yaxis()
        ax_main.invert_xaxis()
    ax_main.legend()
    ax_main.grid(True)

    ax_histx.set_yscale('log')
    ax_histy.set_xscale('log')

    # Quitar etiquetas redundantes
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.show()

# Ejecutar las gráficas optimizadas
#plot_similarity_metric('wasserstein', r'$W_1(B_F, B_F)$', r'$W_1(B_F, B_R)$')
#plot_similarity_metric('match', r'$Mf(B_F, B_F)$', r'$Mf(B_F, B_R)$', invert=True)
#plot_similarity_metric('crosscov', r'$k(B_F, B_F)$', r'$k(B_F, B_R)$', invert=True)


def consistency_test(dataset_orig, label_orig, dataset_gen, labels_gen):
    print("Starting consistency test...")
    unique,counts=np.unique(label_orig, return_counts=True)
    print("Classes, counts for original: ")
    print(np.asarray((unique, counts)).T)

    unique_gen,counts_gen=np.unique(labels_gen, return_counts=True)
    print("Classes, counts for generation: ")
    print(np.asarray((unique_gen, counts_gen)).T)
    print("label_orig shape: ",label_orig.shape)  # esto debería mostrar algo como (N,)

    # ---- CONFIGURACIÓN ----
    np.random.seed(42)
    #N = 300  # muestras por clase
    classes = [0,1,2,3,4]
    # Re-run simulation with faster method
    results = {m: {'x': [], 'y': [], 'label': []} for m in ['wasserstein', 'match', 'crosscov']}

    print("dataset_orig shape:", dataset_orig.shape)
    print("label_orig shape:", label_orig.shape)


    for class_label in classes:
        # Filtrar las señales por clase usando las etiquetas
        B_F_class = dataset_gen[np.array(labels_gen) == class_label]
        B_R_class = dataset_orig[label_orig == class_label]
        #B_R_class = dataset_orig[np.array(label_orig) == class_label]

        # Número de señales disponibles para esta clase (por si difieren)
        N_real = min(len(B_F_class), len(B_R_class))

        print(f"Clase: {class_label}, muestras disponibles: {N_real}")

        for i in range(N_real):
            bF = B_F_class[i]
            x_w, _ = compute_similarity_fast(bF, B_F_class, 'wasserstein')
            y_w, _ = compute_similarity_fast(bF, B_R_class, 'wasserstein')

            x_m, _ = compute_similarity_fast(bF, B_F_class, 'match')
            y_m, _ = compute_similarity_fast(bF, B_R_class, 'match')

            x_k, _ = compute_similarity_fast(bF, B_F_class, 'crosscov')
            y_k, _ = compute_similarity_fast(bF, B_R_class, 'crosscov')

            for metric, x, y in zip(['wasserstein', 'match', 'crosscov'], [x_w, x_m, x_k], [y_w, y_m, y_k]):
                results[metric]['x'].append(x)
                results[metric]['y'].append(y)
                results[metric]['label'].append(class_label)

    # Plot all metrics again
    plot_similarity_metric(classes,  results, 'wasserstein', r'$W_1(B_F, B_F)$', r'$W_1(B_F, B_R)$')
    #plot_similarity_metric(classes, results, 'match', r'$Mf(B_F, B_F)$', r'$Mf(B_F, B_R)$', invert=True)
    #plot_similarity_metric(classes, results, 'crosscov', r'$k(B_F, B_F)$', r'$k(B_F, B_R)$', invert=True)
