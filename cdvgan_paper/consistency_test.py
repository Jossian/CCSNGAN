import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import correlate
from matplotlib.gridspec import GridSpec

from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings('ignore')

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

def dtw_distance(x, y):
    """
    Compute Dynamic Time Warping distance between two time series
    """
    n, m = len(x), len(y)
    # Create cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i-1] - y[j-1])
            # Take minimum of three possible paths
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],      # insertion
                                         dtw_matrix[i, j-1],      # deletion
                                         dtw_matrix[i-1, j-1])    # match
    
    return dtw_matrix[n, m]

def dtw_kernel(X, Y, sigma=1.0):
    """
    Compute DTW kernel matrix between sets X and Y
    """
    n, m = len(X), len(Y)
    K = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            dtw_dist = dtw_distance(X[i], Y[j])
            K[i, j] = np.exp(-dtw_dist / (2 * sigma**2))
    
    return K

def multi_scale_rbf_kernel(X, Y, sigmas=[0.1, 1.0, 10.0]):
    """
    Compute multi-scale RBF kernel matrix
    """
    # Flatten signals for distance computation
    X_flat = X.reshape(len(X), -1)
    Y_flat = Y.reshape(len(Y), -1)
    
    # Compute median distance for scaling
    if len(X_flat) > 1 and len(Y_flat) > 1:
        sample_X = X_flat[:min(100, len(X_flat))]
        sample_Y = Y_flat[:min(100, len(Y_flat))]
        distances = cdist(sample_X, sample_Y, metric='euclidean')
        median_dist = np.median(distances[distances > 0])
    else:
        median_dist = 1.0
    
    # Scale sigmas by median distance
    scaled_sigmas = [s * median_dist for s in sigmas]
    
    K_total = np.zeros((len(X), len(Y)))
    for sigma in scaled_sigmas:
        gamma = 1.0 / (2 * sigma**2)
        K = rbf_kernel(X_flat, Y_flat, gamma=gamma)
        K_total += K
    
    return K_total / len(scaled_sigmas)

def compute_mmd(X, Y, kernel_func, **kernel_params):
    """
    Compute Maximum Mean Discrepancy between two datasets
    """
    n, m = len(X), len(Y)
    
    if n == 0 or m == 0:
        return np.inf
    
    # Compute kernel matrices
    K_XX = kernel_func(X, X, **kernel_params)
    K_YY = kernel_func(Y, Y, **kernel_params)
    K_XY = kernel_func(X, Y, **kernel_params)
    
    # Compute MMD^2
    mmd_squared = (np.sum(K_XX) / (n * n) + 
                   np.sum(K_YY) / (m * m) - 
                   2 * np.sum(K_XY) / (n * m))
    
    return np.sqrt(max(0, mmd_squared))

def compute_mmd_metrics(dataset_orig, label_orig, dataset_gen, labels_gen, classes):
    """
    Compute MMD with different kernels and evaluation schemes
    """
    results = {
        'dtw': {
            'class_conditional': {},
            'overall': 0,
            'cross_class': {}
        },
        'multi_rbf': {
            'class_conditional': {},
            'overall': 0,
            'cross_class': {}
        }
    }
    
    # Class-conditional MMD
    print("\n=== Class-conditional MMD ===")
    for class_label in classes:
        orig_class = dataset_orig[label_orig == class_label]
        gen_class = dataset_gen[np.array(labels_gen) == class_label]
        
        if len(orig_class) > 0 and len(gen_class) > 0:
            # DTW MMD
            mmd_dtw = compute_mmd(orig_class, gen_class, dtw_kernel, sigma=1.0)
            results['dtw']['class_conditional'][class_label] = mmd_dtw
            
            # Multi-scale RBF MMD
            mmd_rbf = compute_mmd(orig_class, gen_class, multi_scale_rbf_kernel)
            results['multi_rbf']['class_conditional'][class_label] = mmd_rbf
            
            print(f"Class {class_label}: DTW-MMD = {mmd_dtw:.4f}, Multi-RBF-MMD = {mmd_rbf:.4f}")
    
    # Overall MMD
    print("\n=== Overall MMD ===")
    mmd_dtw_overall = compute_mmd(dataset_orig, dataset_gen, dtw_kernel, sigma=1.0)
    mmd_rbf_overall = compute_mmd(dataset_orig, dataset_gen, multi_scale_rbf_kernel)
    
    results['dtw']['overall'] = mmd_dtw_overall
    results['multi_rbf']['overall'] = mmd_rbf_overall
    
    print(f"Overall: DTW-MMD = {mmd_dtw_overall:.4f}, Multi-RBF-MMD = {mmd_rbf_overall:.4f}")
    
    # Cross-class MMD
    print("\n=== Cross-class MMD ===")
    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i < j:  # Only compute upper triangle
                orig_class1 = dataset_orig[label_orig == class1]
                orig_class2 = dataset_orig[label_orig == class2]
                gen_class1 = dataset_gen[np.array(labels_gen) == class1]
                gen_class2 = dataset_gen[np.array(labels_gen) == class2]
                
                if len(orig_class1) > 0 and len(orig_class2) > 0 and len(gen_class1) > 0 and len(gen_class2) > 0:
                    # Original cross-class MMD
                    mmd_orig = compute_mmd(orig_class1, orig_class2, multi_scale_rbf_kernel)
                    # Generated cross-class MMD
                    mmd_gen = compute_mmd(gen_class1, gen_class2, multi_scale_rbf_kernel)
                    
                    cross_class_key = f"{class1}_{class2}"
                    results['dtw']['cross_class'][cross_class_key] = abs(mmd_orig - mmd_gen)
                    results['multi_rbf']['cross_class'][cross_class_key] = abs(mmd_orig - mmd_gen)
                    
                    print(f"Classes {class1}-{class2}: Orig-MMD = {mmd_orig:.4f}, Gen-MMD = {mmd_gen:.4f}, Diff = {abs(mmd_orig - mmd_gen):.4f}")
    
    return results

def plot_mmd_results(mmd_results, classes):
    """
    Generate comprehensive plots for MMD results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Class-conditional MMD comparison
    dtw_class_values = [mmd_results['dtw']['class_conditional'].get(c, 0) for c in classes]
    rbf_class_values = [mmd_results['multi_rbf']['class_conditional'].get(c, 0) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, dtw_class_values, width, label='DTW-MMD', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x + width/2, rbf_class_values, width, label='Multi-RBF-MMD', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('MMD Value')
    axes[0, 0].set_title('Class-conditional MMD')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Overall MMD comparison
    overall_values = [mmd_results['dtw']['overall'], mmd_results['multi_rbf']['overall']]
    kernel_names = ['DTW-MMD', 'Multi-RBF-MMD']
    
    bars = axes[0, 1].bar(kernel_names, overall_values, color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0, 1].set_ylabel('MMD Value')
    axes[0, 1].set_title('Overall MMD Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, overall_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom')
    
    # Cross-class MMD heatmap
    cross_class_matrix = np.zeros((len(classes), len(classes)))
    for key, value in mmd_results['multi_rbf']['cross_class'].items():
        class1, class2 = map(int, key.split('_'))
        i, j = classes.index(class1), classes.index(class2)
        cross_class_matrix[i, j] = value
        cross_class_matrix[j, i] = value  # Make symmetric
    
    im = axes[0, 2].imshow(cross_class_matrix, cmap='viridis', aspect='auto')
    axes[0, 2].set_title('Cross-class MMD Differences')
    axes[0, 2].set_xlabel('Class')
    axes[0, 2].set_ylabel('Class')
    axes[0, 2].set_xticks(range(len(classes)))
    axes[0, 2].set_yticks(range(len(classes)))
    axes[0, 2].set_xticklabels(classes)
    axes[0, 2].set_yticklabels(classes)
    plt.colorbar(im, ax=axes[0, 2])
    
    # MMD distribution by class (violin plot)
    class_data = []
    kernel_data = []
    mmd_data = []
    
    for class_label in classes:
        if class_label in mmd_results['dtw']['class_conditional']:
            class_data.append(class_label)
            kernel_data.append('DTW')
            mmd_data.append(mmd_results['dtw']['class_conditional'][class_label])
        
        if class_label in mmd_results['multi_rbf']['class_conditional']:
            class_data.append(class_label)
            kernel_data.append('Multi-RBF')
            mmd_data.append(mmd_results['multi_rbf']['class_conditional'][class_label])
    
    if class_data:
        import pandas as pd
        df = pd.DataFrame({
            'Class': class_data,
            'Kernel': kernel_data,
            'MMD': mmd_data
        })
        
        # Create violin plot
        for i, kernel in enumerate(['DTW', 'Multi-RBF']):
            kernel_data_subset = df[df['Kernel'] == kernel]
            if not kernel_data_subset.empty:
                positions = [c + i*0.4 - 0.2 for c in kernel_data_subset['Class']]
                axes[1, 0].scatter(positions, kernel_data_subset['MMD'], 
                                 alpha=0.7, s=60, label=kernel if i == 0 else "")
        
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('MMD Value')
        axes[1, 0].set_title('MMD Values by Class and Kernel')
        axes[1, 0].set_xticks(classes)
        axes[1, 0].legend(['DTW', 'Multi-RBF'])
        axes[1, 0].grid(True, alpha=0.3)
    
    # Kernel comparison scatter plot
    dtw_values = list(mmd_results['dtw']['class_conditional'].values())
    rbf_values = list(mmd_results['multi_rbf']['class_conditional'].values())
    
    if dtw_values and rbf_values:
        axes[1, 1].scatter(dtw_values, rbf_values, alpha=0.7, s=100, c=classes[:len(dtw_values)], cmap='tab10')
        
        # Add diagonal line
        min_val = min(min(dtw_values), min(rbf_values))
        max_val = max(max(dtw_values), max(rbf_values))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        
        axes[1, 1].set_xlabel('DTW-MMD')
        axes[1, 1].set_ylabel('Multi-RBF-MMD')
        axes[1, 1].set_title('Kernel Comparison (Class-conditional)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add class labels
        for i, (x, y) in enumerate(zip(dtw_values, rbf_values)):
            axes[1, 1].annotate(f'C{classes[i]}', (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
    
    # MMD summary statistics
    axes[1, 2].axis('off')
    
    # Create summary text
    summary_text = "MMD Summary Statistics\n" + "="*25 + "\n\n"
    
    summary_text += f"Overall MMD:\n"
    summary_text += f"  DTW-MMD: {mmd_results['dtw']['overall']:.4f}\n"
    summary_text += f"  Multi-RBF-MMD: {mmd_results['multi_rbf']['overall']:.4f}\n\n"
    
    if dtw_values:
        summary_text += f"Class-conditional MMD:\n"
        summary_text += f"  DTW - Mean: {np.mean(dtw_values):.4f}, Std: {np.std(dtw_values):.4f}\n"
        summary_text += f"  Multi-RBF - Mean: {np.mean(rbf_values):.4f}, Std: {np.std(rbf_values):.4f}\n\n"
    
    if mmd_results['multi_rbf']['cross_class']:
        cross_values = list(mmd_results['multi_rbf']['cross_class'].values())
        summary_text += f"Cross-class MMD differences:\n"
        summary_text += f"  Mean: {np.mean(cross_values):.4f}\n"
        summary_text += f"  Std: {np.std(cross_values):.4f}\n"
        summary_text += f"  Max: {np.max(cross_values):.4f}\n"
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
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
    print("label_orig shape: ",label_orig.shape)

    # ---- CONFIGURACIÓN ----
    np.random.seed(42)
    classes = [0,1,2,3,4]
    
    print("dataset_orig shape:", dataset_orig.shape)
    print("label_orig shape:", label_orig.shape)

    # Original similarity tests
    results = {m: {'x': [], 'y': [], 'label': []} for m in ['wasserstein', 'match', 'crosscov']}

    for class_label in classes:
        # Filtrar las señales por clase usando las etiquetas
        B_F_class = dataset_gen[np.array(labels_gen) == class_label]
        B_R_class = dataset_orig[label_orig == class_label]

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

    # Plot original similarity metrics
    plot_similarity_metric(classes, results, 'wasserstein', r'$W_1(B_F, B_F)$', r'$W_1(B_F, B_R)$')
    plot_similarity_metric(classes, results, 'match', r'$Mf(B_F, B_F)$', r'$Mf(B_F, B_R)$')
    plot_similarity_metric(classes, results, 'crosscov', r'$k(B_F, B_F)$', r'$k(B_F, B_R)$')

    # NEW: MMD Analysis
    print("\n" + "="*50)
    print("MAXIMUM MEAN DISCREPANCY ANALYSIS")
    print("="*50)
    
    mmd_results = compute_mmd_metrics(dataset_orig, label_orig, dataset_gen, labels_gen, classes)
    plot_mmd_results(mmd_results, classes)
    
    return results, mmd_results