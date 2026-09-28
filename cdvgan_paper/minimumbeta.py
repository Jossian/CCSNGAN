import h5py
import requests
from io import BytesIO
import numpy as np

url = "https://zenodo.org/record/201145/files/GWdatabase.h5?download=1"

# Descargar el archivo en memoria
r = requests.get(url)
r.raise_for_status()
file_in_memory = BytesIO(r.content)

# Leer beta1_IC_b
with h5py.File(file_in_memory, "r") as f:
    beta_all = f["reduced_data"]["beta1_IC_b"][:]

    # Filtrar valores válidos y beta > 0
    beta_valid = beta_all[beta_all > 0]

    # Encontrar el mínimo
    beta_min = np.min(beta_valid)

print(f"Valor mínimo de beta1_IC_b: {beta_min}")