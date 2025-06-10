import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import correlate
from matplotlib.gridspec import GridSpec

from scipy.spatial.distance import cdist
from fastdtw import fastdtw
from sklearn.metrics.pairwise import rbf_kernel

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

    colors = {
        0: 'royalblue',
        1: 'darkorange',
        2: 'forestgreen',
        3: 'crimson',
        4: 'mediumvioletred'
    }

    for cls in classes:
        # Si cls es one-hot encoding, conviértelo a número:
        if isinstance(cls, (list, np.ndarray)):
            cls_numeric = int(np.argmax(cls))
        else:
            cls_numeric = int(cls)

        cls_mask = np.array(results[metric_name]['label']) == cls
        x = np.array(results[metric_name]['x'])[cls_mask]
        y = np.array(results[metric_name]['y'])[cls_mask]

        color = colors.get(cls_numeric, 'gray')
        ax_main.scatter(x, y, label=cls_numeric, alpha=0.5, s=10, color=color)

        # Ajuste lineal
        slope, intercept, *_ = linregress(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        ax_main.plot(x_line, slope * x_line + intercept, color=color,
                     label=fr"{cls_numeric}: $y = {slope:.2f}x + {intercept:.2f}$")

        # Histograma superior
        ax_histx.hist(x, bins=50, color=color, alpha=0.6)
        x_mean, x_std = np.mean(x), np.std(x)
        ax_histx.axvline(x_mean - 6 * x_std, linestyle='--', color=color, alpha=0.6)
        ax_histx.axvline(x_mean + 6 * x_std, linestyle='--', color=color, alpha=0.6)

        # Histograma lateral
        ax_histy.hist(y, bins=50, color=color, orientation='horizontal', alpha=0.6)
        y_mean, y_std = np.mean(y), np.std(y)
        ax_histy.axhline(y_mean - 6 * y_std, linestyle='--', color=color, alpha=0.6)
        ax_histy.axhline(y_mean + 6 * y_std, linestyle='--', color=color, alpha=0.6)

    # Ejes y estilo
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)

    # Título general de la figura (no se superpone al histograma superior)
    fig.suptitle(f"Similarity: {metric_name.capitalize()}", y=0.95, fontsize=14)

    if invert:
        ax_main.invert_yaxis()
        ax_main.invert_xaxis()
    ax_main.legend()
    ax_main.grid(True)

    ax_histx.set_yscale('log')
    ax_histy.set_xscale('log')

    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    fig = plt.figure(constrained_layout=True)  # deja espacio para el título arriba
    plt.show()

# Ejecutar las gráficas optimizadas
#plot_similarity_metric('wasserstein', r'$W_1(B_F, B_F)$', r'$W_1(B_F, B_R)$')
#plot_similarity_metric('match', r'$Mf(B_F, B_F)$', r'$Mf(B_F, B_R)$', invert=True)
#plot_similarity_metric('crosscov', r'$k(B_F, B_F)$', r'$k(B_F, B_R)$', invert=True)



##################################MDD FUNCTIONS######################################

def compute_dtw_distance_matrix(X, Y):
    dist_matrix = np.zeros((len(X), len(Y)))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            dist, _ = fastdtw(x, y)
            dist_matrix[i, j] = dist
    return dist_matrix

def multi_scale_rbf_kernel(D, sigmas=[0.1, 1.0, 10.0]):
    """Multi-scale RBF kernel from distance matrix"""
    K = np.zeros_like(D)
    for sigma in sigmas:
        K += np.exp(-D**2 / (2 * sigma**2))
    return K

def compute_mmd(K_xx, K_yy, K_xy):
    """Biased estimator of MMD^2"""
    m = K_xx.shape[0]
    n = K_yy.shape[0]
    return (np.sum(K_xx) / (m * m) +
            np.sum(K_yy) / (n * n) -
            2 * np.sum(K_xy) / (m * n))

######################################################################################

def consistency_test(dataset_orig, label_orig, dataset_gen, labels_gen):
    print("Starting consistency test...")

    unique, counts = np.unique(label_orig, return_counts=True)
    print("Original class distribution:\n", np.asarray((unique, counts)).T)
    unique_gen, counts_gen = np.unique(labels_gen, return_counts=True)
    print("Generated class distribution:\n", np.asarray((unique_gen, counts_gen)).T)

    np.random.seed(42)
    classes = np.unique(label_orig)
    results = {m: {'x': [], 'y': [], 'label': []} for m in ['wasserstein', 'match', 'crosscov']}
    mmd_results = {'class_conditional': {}, 'overall': None, 'cross_class': {}}

    # Overall MMD
    print("\nComputing Overall MMD using DTW + Multi-Scale RBF...")
    D_oo = compute_dtw_distance_matrix(dataset_orig, dataset_orig)
    D_gg = compute_dtw_distance_matrix(dataset_gen, dataset_gen)
    D_og = compute_dtw_distance_matrix(dataset_orig, dataset_gen)

    K_oo = multi_scale_rbf_kernel(D_oo)
    K_gg = multi_scale_rbf_kernel(D_gg)
    K_og = multi_scale_rbf_kernel(D_og)

    mmd_results['overall'] = compute_mmd(K_oo, K_gg, K_og)

    # Class-conditional MMD
    print("\nComputing Class-conditional MMD...")
    for c in classes:
        X = dataset_orig[label_orig == c]
        Y = dataset_gen[labels_gen == c]

        if len(X) < 2 or len(Y) < 2:
            print(f"Skipping class {c} (too few samples)")
            continue

        D_xx = compute_dtw_distance_matrix(X, X)
        D_yy = compute_dtw_distance_matrix(Y, Y)
        D_xy = compute_dtw_distance_matrix(X, Y)

        K_xx = multi_scale_rbf_kernel(D_xx)
        K_yy = multi_scale_rbf_kernel(D_yy)
        K_xy = multi_scale_rbf_kernel(D_xy)

        mmd = compute_mmd(K_xx, K_yy, K_xy)
        mmd_results['class_conditional'][c] = mmd

    # Cross-class MMD
    print("\nComputing Cross-class MMD...")
    for i in classes:
        for j in classes:
            if i >= j: continue
            Xi = dataset_orig[label_orig == i]
            Yj = dataset_gen[labels_gen == j]
            if len(Xi) < 2 or len(Yj) < 2:
                continue

            D_ij = compute_dtw_distance_matrix(Xi, Yj)
            K_ij = multi_scale_rbf_kernel(D_ij)
            mmd_cross = np.mean(K_ij)
            mmd_results['cross_class'][(i, j)] = mmd_cross

    # Plotting Class-conditional MMD
    plt.figure(figsize=(8, 5))
    keys = list(mmd_results['class_conditional'].keys())
    values = [mmd_results['class_conditional'][k] for k in keys]
    plt.bar(keys, values)
    plt.title("Class-conditional MMD using DTW + Multi-scale RBF")
    plt.xlabel("Class")
    plt.ylabel("MMD²")
    plt.grid(True)
    plt.show()

    # Plotting Cross-class MMD heatmap
    cross_class_pairs = list(mmd_results['cross_class'].keys())
    if cross_class_pairs:
        matrix = np.zeros((len(classes), len(classes)))
        for (i, j), val in mmd_results['cross_class'].items():
            matrix[i, j] = val
            matrix[j, i] = val  # symmetry
        plt.figure(figsize=(6, 5))
        plt.imshow(matrix, interpolation='nearest', cmap='viridis')
        plt.colorbar(label="MMD Cross-class")
        plt.title("Cross-class MMD Heatmap")
        plt.xticks(classes)
        plt.yticks(classes)
        plt.xlabel("Class")
        plt.ylabel("Class")
        plt.show()

    # Print Overall MMD
    print(f"\nOverall MMD: {mmd_results['overall']:.4f}")