import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import random
import sys
from pathlib import Path
import requests
from io import BytesIO

url = "https://zenodo.org/record/201145/files/GWdatabase.h5?download=1"

# Descargar el archivo en memoria
r = requests.get(url)
file_in_memory = BytesIO(r.content)

# === Rutas del proyecto (Ajuste necesario para notebooks o scripts) ===
# NOTA: En un script o notebook, la ruta_proyecto debe definirse manualmente
# si no existe un archivo __file__. Aquí lo ajustaremos para ser independiente.
try:
    ruta_actual = Path(__file__).resolve()
    ruta_proyecto = ruta_actual.parents[0]
except NameError:
    # Usar el directorio actual si __file__ no está definido (e.g., Jupyter Notebook)
    ruta_proyecto = Path(".")
    
ruta_data = ruta_proyecto / "data"
ruta_data.mkdir(exist_ok=True) # Crea el directorio 'data' si no existe
    
if str(ruta_proyecto) not in sys.path:
    sys.path.insert(0, str(ruta_proyecto))

# === Parámetros de preprocesamiento ===
ALPHA_TUKEY = 0.1
ORDER_BUTTER = 10
ATTENUATION = 0.25
N_SAMPLES_FINAL = 256
WINDOW_BEFORE_MS = 12.8e-3  # 12.8 ms
WINDOW_AFTER_MS = 49.55e-3  # 49.55 ms

# === Inicialización ===
signals_processed = []
labels = []
# 🌟 VARIABLE PARA GUARDAR EL TIEMPO DESPUÉS DE LA INTERPOLACIÓN 🌟
time_interpolated = None 

with h5py.File(file_in_memory, "r") as f:
    waveforms_group = f["waveforms"]
    reduced_data_group = f["reduced_data"]
    waveform_names = list(waveforms_group.keys())
    print(f"Número total de señales: {len(waveform_names)}")

    # Extraer todos los parámetros de reduced_data
    beta_all = reduced_data_group["beta1_IC_b"][:]
    EOS_all = reduced_data_group["EOS"][:]
    A_all = reduced_data_group["A(km)"][:]
    omega0_all = reduced_data_group["omega_0(rad|s)"][:]
    tbounce_all = reduced_data_group["tbounce(s)"][:] # Extraer tbounce aquí

    for i, name in enumerate(waveform_names):
        grp = waveforms_group[name]
        time = np.array(grp["t-tb(s)"][:])      # segundos, centrado en bounce
        hD = np.array(grp["strain*dist(cm)"][:])    # strain * distancia (cm)

        # --- Tomar parámetros de reduced_data ---
        beta1_IC_b = beta_all[i]
        EOS = EOS_all[i].decode("utf-8") if isinstance(EOS_all[i], bytes) else EOS_all[i]
        A = A_all[i]
        omega0 = omega0_all[i]
        tb = tbounce_all[i] # Usar el array extraído

        # --- Filtrar señales con beta1_IC_b <= 0 ---
        if beta1_IC_b <= 0:
            continue

        # --- Aplicar ventana Tukey ---
        window = tukey(len(hD), alpha=ALPHA_TUKEY)
        hD_win = hD * window

        # --- Recorte temporal alrededor del bounce ---
        mask = (time >= -WINDOW_BEFORE_MS) & (time <= WINDOW_AFTER_MS)
        if not np.any(mask):
            continue

        time_crop = time[mask]
        h_crop = hD_win[mask]

        # --- Interpolación lineal a N_SAMPLES_FINAL puntos ---
        t_new = np.linspace(time_crop[0], time_crop[-1], N_SAMPLES_FINAL)
        
        # 🌟 GUARDAR EL ARRAY DE TIEMPO INTERPOLADO UNA SOLA VEZ 🌟
        if time_interpolated is None:
            time_interpolated = t_new
            
        interp_func = interp1d(time_crop, h_crop, kind="linear", fill_value="extrapolate")
        h_interp = interp_func(t_new)

        # --- Filtro Butterworth pasa bajas ---
        b, a = butter(ORDER_BUTTER, ATTENUATION) # Usar la variable ATTENUATION
        h_filt = filtfilt(b, a, h_interp)

        # --- Atenuación y Normalización ---
        h_final = h_filt * ATTENUATION
        
        mean_val = np.mean(h_final)
        std_val = np.std(h_final)
        if std_val < 1e-8:
            continue
        h_norm = (h_final - mean_val) / std_val

        signals_processed.append(h_norm)

        # --- Guardar etiquetas ---
        labels.append({
            "name": name,
            "EOS": EOS,
            "A(km)": A,
            "omega_0(rad/s)": omega0,
            "beta1_IC_b": beta1_IC_b,
            "tbounce_s": tb,     
            "ndata": len(time)
        })

# === Convertir a DataFrames y guardar ===
signals_df = pd.DataFrame(signals_processed)
labels_df = pd.DataFrame(labels)

# 🌟 GUARDAR EL ARRAY DE TIEMPO INTERPOLADO EN UN ARCHIVO SEPARADO 🌟
# Lo guardamos en segundos, ya que así está 'time_interpolated'
time_df = pd.DataFrame({"time_s": time_interpolated}) 

signals_df.to_csv(ruta_data / "signals_preprocessed.csv", index=False)
labels_df.to_csv(ruta_data / "labels.csv", index=False)
time_df.to_csv(ruta_data / "time_interpolated.csv", index=False)

print("\n✅ Procesamiento completado.")
print(f"Señales procesadas: {len(signals_df)}")
print(f"Archivos guardados en: {ruta_data}")
print("Archivos: signals_preprocessed.csv, labels.csv, time_interpolated.csv")

# ----------------------------------------------------------------------
# === Graficar señales aleatorias (Usando el array de tiempo guardado) ===
# ----------------------------------------------------------------------

if time_interpolated is not None:
    # Convertir el tiempo de segundos a milisegundos para la gráfica
    t_plot_ms = time_interpolated * 1e3 

    num_examples = min(5, len(signals_df))
    if num_examples > 0:
        idx_random = random.sample(range(len(signals_df)), num_examples)
        
        plt.figure(figsize=(12, 6 * num_examples // 2))
        
        for i, idx in enumerate(idx_random, 1):
            ax = plt.subplot(num_examples, 1, i)
            
            # Usar el array de tiempo guardado
            ax.plot(t_plot_ms, signals_df.iloc[idx].values.astype(float), lw=1)
            
            label = labels_df.iloc[idx]
            
            ax.set_title(f"{label['name']} | EOS={label['EOS']} | A={label['A(km)']} km | β={label['beta1_IC_b']:.3f}", fontsize=9)
            ax.set_xlabel("Tiempo (ms)")
            ax.set_ylabel("Amplitud normalizada")
            ax.grid(True, alpha=0.3)
            
            # Línea vertical para el centro del bounce (t-tb=0)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, label='t = 0')

            # 2. Línea vertical en t = 0 + tb (Rebote del núcleo)
            # Asumiendo que 'time' representa el tiempo después del inicio del colapso
            # y que tbounce (tb) ya está en unidades de tiempo consistentes con 'time'.
            ax.axvline(
                x=label['tbounce_s'], 
                color='red', 
                linestyle='-', 
                linewidth=1.5, 
                label=f't = $t_b$ ({label["tbounce_s"]:.3f})'
            )

            # Añadir leyenda
            ax.legend(loc='upper right', fontsize=7)

        plt.tight_layout()
        plt.show()

# === Verificación rápida ===
print("\n--- Verificación rápida ---")
print("time_df info:")
print(time_df.info())
print("Ejemplo de tiempo (primeros 5 valores en segundos):")
print(time_df.head())