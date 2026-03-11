import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

ruta_actual = Path(__file__).resolve()
ruta_proyecto = ruta_actual.parents[0]
if str(ruta_proyecto) not in sys.path:
    sys.path.insert(0, str(ruta_proyecto))

# Cargar datos originales
data_orig = np.loadtxt(f'{ruta_proyecto}/data/ConditionalSignals_orig_beta.csv', delimiter=',')
class_array_orig = np.loadtxt(f'{ruta_proyecto}/data/ConditionalLabels_orig_beta.csv', delimiter=',')
class_array_orig = class_array_orig - 1

# Cargar datos aumentados
data_aug = np.loadtxt(f'{ruta_proyecto}/data/ConditionalSignals_aug_beta.csv', delimiter=',')
class_array_aug = np.loadtxt(f'{ruta_proyecto}/data/ConditionalLabels_aug_beta.csv', delimiter=',')

# Verificar si las clases aumentadas necesitan ajuste
print(f"Clases originales únicas: {np.unique(class_array_orig)}")
print(f"Clases aumentadas únicas: {np.unique(class_array_aug)}")

# Si las clases aumentadas están en rango [1, n] en lugar de [0, n-1], ajustar
if np.min(class_array_aug) == 1:
    class_array_aug = class_array_aug - 1
    print("Ajustando clases aumentadas para que coincidan con originales")

# Obtener las clases únicas
unique_classes = np.unique(class_array_orig)

# Colores para las gráficas
color_orig = '#0066CC'  # Azul
color_aug = '#CC0066'   # Rojo/Rosa

for i, clase in enumerate(unique_classes):
    # Crear una figura individual para cada clase
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Filtrar señales por clase - ORIGINALES
    indices_orig = np.where(class_array_orig == clase)[0]
    signals_orig = data_orig[indices_orig]
    
    # Filtrar señales por clase - AUMENTADAS
    indices_aug = np.where(class_array_aug == clase)[0]
    signals_aug = data_aug[indices_aug]
    
    # Verificar que hay datos
    if len(signals_orig) == 0:
        print(f"Advertencia: No hay señales originales para clase {int(clase)}")
        continue
    if len(signals_aug) == 0:
        print(f"Advertencia: No hay señales aumentadas para clase {int(clase)}")
        continue
    
    # Calcular mediana y percentiles para originales
    median_orig = np.median(signals_orig, axis=0)
    percentile_25_orig = np.percentile(signals_orig, 25, axis=0)
    percentile_75_orig = np.percentile(signals_orig, 75, axis=0)
    
    # Calcular mediana y percentiles para aumentadas
    median_aug = np.median(signals_aug, axis=0)
    percentile_25_aug = np.percentile(signals_aug, 25, axis=0)
    percentile_75_aug = np.percentile(signals_aug, 75, axis=0)
    
    # Crear eje de tiempo (asumiendo que la señal representa segundos)
    time_axis = np.linspace(-0.02, 0.05, len(median_orig))
    
    # Graficar señales originales
    ax.fill_between(time_axis, percentile_25_orig, percentile_75_orig, 
                     color=color_orig, alpha=0.3)
    ax.plot(time_axis, median_orig, color=color_orig, linewidth=2, 
            label=f'Original Median (n={len(signals_orig)})')
    
    # Graficar señales aumentadas
    ax.fill_between(time_axis, percentile_25_aug, percentile_75_aug, 
                     color=color_aug, alpha=0.3)
    ax.plot(time_axis, median_aug, color=color_aug, linewidth=2, 
            label=f'All Median (n={len(signals_aug)})')
    
    # Configurar etiquetas y títulos
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Normalized amplitude', fontsize=11)
    ax.set_ylim(-0.9, 0.4)
    ax.set_title(f'Comparison: Original vs All - Class {int(clase)}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('comparison_original_vs_augmented_all_classes.png', dpi=300, bbox_inches='tight')
plt.show()

# Imprimir estadísticas
print("\n" + "="*70)
print("ESTADÍSTICAS DE DISTRIBUCIÓN POR CLASE")
print("="*70)
for clase in unique_classes:
    n_orig = np.sum(class_array_orig == clase)
    n_aug = np.sum(class_array_aug == clase)
    print(f"\nClase {int(clase)}:")
    print(f"  Original: {n_orig} señales")
    print(f"  Aumentado: {n_aug} señales")
    print(f"  Incremento: {n_aug - n_orig} señales ({((n_aug/n_orig - 1)*100):.1f}%)")
print("="*70)
