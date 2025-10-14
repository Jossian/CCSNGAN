import h5py
import numpy as np
import pandas as pd
from scipy.signal import tukey, butter, filtfilt
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

# === Rutas del proyecto ===
ruta_actual = Path(__file__).resolve()
ruta_proyecto = ruta_actual.parents[0]
if str(ruta_proyecto) not in sys.path:
    sys.path.insert(0, str(ruta_proyecto))

# === Parámetros de preprocesamiento ===
ALPHA_TUKEY = 0.1
ORDER_BUTTER = 10
ATTENUATION = 0.25
N_SAMPLES_FINAL = 256
WINDOW_BEFORE_MS = 12.8e-3  # 12.8 ms
WINDOW_AFTER_MS = 49.55e-3  # 49.55 ms

# === Abrir el archivo ===
#fname = ruta_proyecto / "data" / "GWdatabase.h5"

signals_processed = []
labels = []

"""with h5py.File(fname, "r") as f:
"""
f=file_in_memory
waveforms_group = f["waveforms"]
reduced_data_group = f["reduced_data"]
waveform_names = list(waveforms_group.keys())
print(f"Número total de señales: {len(waveform_names)}")

# Extraer todos los parámetros de reduced_data
beta_all = reduced_data_group["beta1_IC_b"][:]
EOS_all = reduced_data_group["EOS"][:]
A_all = reduced_data_group["A(km)"][:]
omega0_all = reduced_data_group["omega_0(rad|s)"][:]

for i, name in enumerate(waveform_names):
    grp = waveforms_group[name]
    time = np.array(grp["t-tb(s)"][:])           # segundos, centrado en bounce
    hD = np.array(grp["strain*dist(cm)"][:])     # strain * distancia (cm)

    # --- Tomar parámetros de reduced_data ---
    beta1_IC_b = beta_all[i]
    EOS = EOS_all[i].decode("utf-8") if isinstance(EOS_all[i], bytes) else EOS_all[i]
    A = A_all[i]
    omega0 = omega0_all[i]

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
    interp_func = interp1d(time_crop, h_crop, kind="linear", fill_value="extrapolate")
    h_interp = interp_func(t_new)

    # --- Filtro Butterworth pasa bajas ---
    b, a = butter(ORDER_BUTTER, 0.25)
    h_filt = filtfilt(b, a, h_interp)

    # --- Atenuación ---
    h_final = h_filt * ATTENUATION

    # --- Normalización ---
    mean_val = np.mean(h_final)
    std_val = np.std(h_final)
    if std_val < 1e-8:
        continue
    h_norm = (h_final - mean_val) / std_val

    signals_processed.append(h_norm)

    tb = reduced_data_group["tbounce(s)"][i]  # tiempo del core bounce en segundos

    labels.append({
        "name": name,
        "EOS": EOS,
        "A(km)": A,
        "omega_0(rad/s)": omega0,
        "beta1_IC_b": beta1_IC_b,
        "tbounce_s": tb,       # <-- agregado
        "ndata": len(time)
    })

# === Convertir a DataFrames y guardar ===
signals_df = pd.DataFrame(signals_processed)
labels_df = pd.DataFrame(labels)

signals_df.to_csv(ruta_proyecto / "data" / "signals_preprocessed.csv", index=False)
labels_df.to_csv(ruta_proyecto / "data" / "labels.csv", index=False)

print("\n✅ Procesamiento completado.")
print(f"Señales procesadas: {len(signals_df)}")
print("Archivos guardados: signals_preprocessed.csv, labels.csv")

# === Graficar señales aleatorias ===
num_examples = min(5, len(signals_df))
idx_random = random.sample(range(len(signals_df)), num_examples)
t_plot = np.linspace(-WINDOW_BEFORE_MS*1e3, WINDOW_AFTER_MS*1e3, N_SAMPLES_FINAL)  # tiempo en ms

plt.figure(figsize=(12, 6))
for i, idx in enumerate(idx_random, 1):
    plt.subplot(num_examples, 1, i)
    plt.plot(t_plot, signals_df.iloc[idx].values.astype(float), lw=1)
    label = labels_df.iloc[idx]
    plt.title(f"{label['name']} | EOS={label['EOS']} | A={label['A(km)']} km | β={label['beta1_IC_b']:.3f}", fontsize=9)
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Amplitud normalizada")
    plt.grid(True, alpha=0.3)
    plt.axvline(0, color='r', linestyle='--', lw=0.8, alpha=0.7, label="tb=0 ms")
    plt.legend(loc='upper right', fontsize=7)

plt.tight_layout()
plt.show()

# === Verificación rápida ===
print("\n--- Verificación rápida ---")
print("signals_df info:")
print(signals_df.info())
print("Ejemplo de fila (primeros 10 valores):")
print(signals_df.iloc[0, :10])





