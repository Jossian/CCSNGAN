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
        print("DEBUG SHAPES", metric_name, "x:", np.array(x).shape, "y:", np.array(y).shape)

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
    total = len(X) * len(Y)
    count = 0

    print(f"🔁 Calculando matriz de distancias DTW ({len(X)} x {len(Y)})...", flush=True)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            dist, _ = fastdtw(x, y)
            dist_matrix[i, j] = dist
            count += 1

            # Imprimir progreso cada 5%
            if count % max(total // 20, 1) == 0:
                print(f"  Progreso: {100 * count // total}% ({count}/{total})", flush=True)

    print("✅ Matriz de distancias DTW completa.\n", flush=True)
    return dist_matrix

def multi_scale_rbf_kernel(D, sigmas=[0.1, 1.0, 10.0]):
    """Multi-scale RBF kernel from distance matrix"""
    print(f"🧮 Aplicando kernel RBF multi-escala con sigmas = {sigmas}", flush=True)
    K = np.zeros_like(D)
    for sigma in sigmas:
        print(f"  ➤ Procesando sigma = {sigma}", flush=True)
        K += np.exp(-D**2 / (2 * sigma**2))
    print("✅ Kernel RBF calculado.\n", flush=True)
    return K

def compute_mmd(K_xx, K_yy, K_xy):
    """Biased estimator of MMD^2"""
    print("📏 Calculando MMD...", flush=True)
    m = K_xx.shape[0]
    n = K_yy.shape[0]
    mmd2 = (np.sum(K_xx) / (m * m) +
            np.sum(K_yy) / (n * n) -
            2 * np.sum(K_xy) / (m * n))
    print(f"✅ MMD^2 calculado: {mmd2:.6f}\n", flush=True)
    return mmd2




def plot_histograms(data1, data2, data3, labels=None, bins=50, figsize=(15, 4), colors=None):
    """
    Plotea tres histogramas como subplots.

    Parámetros:
    - data1, data2, data3: listas o arrays con los datos.
    - labels: lista de títulos para cada histograma. Por defecto: ["Hist 1", "Hist 2", "Hist 3"]
    - bins: número de bins o lista de bins para los histogramas.
    - figsize: tamaño de la figura (ancho, alto).
    - colors: lista de colores para los histogramas.

    """
    if labels is None:
        labels = ["Hist 1", "Hist 2", "Hist 3"]
    if colors is None:
        colors = ["skyblue", "salmon", "lightgreen"]

    data_list = [data1, data2, data3]
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    for i, ax in enumerate(axs):
        ax.hist(data_list[i], bins=bins, color=colors[i], edgecolor='black')
        ax.set_title(labels[i])
        ax.grid(True)

    plt.tight_layout()
    plt.show()
######################################################################################

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
            bR= B_R_class[i]

            x_w,_  = compute_similarity_fast(bF, B_F_class, 'wasserstein')
            y_w,_  = compute_similarity_fast(bF, B_R_class, 'wasserstein')
            orig_hist_w=compute_similarity_fast(bR, B_R_class, 'wasserstein')

            x_m,_  = compute_similarity_fast(bF, B_F_class, 'match')
            y_m,_  = compute_similarity_fast(bF, B_R_class, 'match')
            orig_hist_m=compute_similarity_fast(bR, B_R_class, 'match')

            x_k,_  = compute_similarity_fast(bF, B_F_class, 'crosscov')
            y_k,_  = compute_similarity_fast(bF, B_R_class, 'crosscov')
            orig_hist_c=compute_similarity_fast(bR, B_R_class, 'crosscov')

            for metric, x, y in zip(['wasserstein', 'match', 'crosscov'], [x_w, x_m, x_k], [y_w, y_m, y_k]):
                results[metric]['x'].append(x)
                results[metric]['y'].append(y)
                results[metric]['label'].append(class_label)

    plot_histograms(orig_hist_w,orig_hist_m,orig_hist_c,labels=classes)
    # Plot all metrics again
    plot_similarity_metric(classes,  results, 'wasserstein', r'$W_1(B_F, B_F)$', r'$W_1(B_F, B_R)$')
    plot_similarity_metric(classes, results, 'match', r'$Mf(B_F, B_F)$', r'$Mf(B_F, B_R)$')
    plot_similarity_metric(classes, results, 'crosscov', r'$k(B_F, B_F)$', r'$k(B_F, B_R)$')

    # --------- ⬇️ AGREGADO: MMD CON DTW Y KERNEL RBF MULTIESCALA ⬇️ ----------
    print("\n🔍 Calculando MMD usando DTW + multi-scale RBF...")
    mmd_results = {'overall': None, 'class_conditional': {}, 'cross_class': {}}

    # Overall MMD
    D_rr = compute_dtw_distance_matrix(dataset_orig, dataset_orig)
    D_ff = compute_dtw_distance_matrix(dataset_gen, dataset_gen)
    D_rf = compute_dtw_distance_matrix(dataset_orig, dataset_gen)

    K_rr = multi_scale_rbf_kernel(D_rr)
    K_ff = multi_scale_rbf_kernel(D_ff)
    K_rf = multi_scale_rbf_kernel(D_rf)

    mmd_results['overall'] = compute_mmd(K_rr, K_ff, K_rf)
    print(f"✅ Overall MMD²: {mmd_results['overall']:.4f}")

    # Class-Conditional MMD
    for c in classes:
        X = dataset_orig[label_orig == c]
        Y = dataset_gen[labels_gen == c]

        ### prueba rápida
        #X = X[:30]
        #Y = Y[:30]
        ###
        if len(X) < 2 or len(Y) < 2:
            print(f"⚠️ Clase {c} omitida por insuficientes muestras.")
            continue
        D_xx = compute_dtw_distance_matrix(X, X)
        D_yy = compute_dtw_distance_matrix(Y, Y)
        D_xy = compute_dtw_distance_matrix(X, Y)
        K_xx = multi_scale_rbf_kernel(D_xx)
        K_yy = multi_scale_rbf_kernel(D_yy)
        K_xy = multi_scale_rbf_kernel(D_xy)
        mmd = compute_mmd(K_xx, K_yy, K_xy)
        mmd_results['class_conditional'][c] = mmd

    # Cross-Class MMD
    for i in classes:
        for j in classes:
            if i >= j: continue
            Xi = dataset_orig[label_orig == i]
            Yj = dataset_gen[labels_gen == j]
            if len(Xi) < 2 or len(Yj) < 2: continue
            D_ij = compute_dtw_distance_matrix(Xi, Yj)
            K_ij = multi_scale_rbf_kernel(D_ij)
            mmd_cross = np.mean(K_ij)
            mmd_results['cross_class'][(i, j)] = mmd_cross

    # 📊 Plots
    # Class-conditional MMD
    keys = list(mmd_results['class_conditional'].keys())
    values = [mmd_results['class_conditional'][k] for k in keys]
    plt.figure(figsize=(8, 5))
    plt.bar(keys, values)
    plt.title("Class-conditional MMD (DTW + RBF)")
    plt.xlabel("Class")
    plt.ylabel("MMD²")
    plt.grid(True)
    plt.show()

    # Cross-class heatmap
    matrix = np.zeros((len(classes), len(classes)))
    for (i, j), val in mmd_results['cross_class'].items():
        i_idx = int(i)
        j_idx = int(j)
        matrix[i_idx, j_idx] = val  
        matrix[j_idx, i_idx] = val  # symmetry
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation='nearest', cmap='viridis')
    plt.colorbar(label="Cross-class similarity (avg. kernel)")
    plt.title("Cross-class MMD Heatmap")
    plt.xticks(classes)
    plt.yticks(classes)
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.show()
    # --------- ⬆️ FIN DEL BLOQUE AGREGADO ⬆️ ----------