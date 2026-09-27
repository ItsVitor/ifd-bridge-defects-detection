import numpy as np
import pandas as pd
import mat73
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
FILENAME = 'data/dano_tipo_5.mat'
FREQUENCIA_HZ = 256
DT = 1.0 / FREQUENCIA_HZ

print(f"--- Processando com MAT73: {FILENAME} ---")

mat = mat73.loadmat(FILENAME)
data = mat[list(mat.keys())[0]]
# MAT73: data geralmente é uma lista de dicionários
num_experimentos = len(data["Accel_X"])

lista_de_dfs = []

for i in tqdm(range(num_experimentos), desc="Experimentos"):
    
    velocidade = data['Velocidade'][i].astype(np.uint8)
    
    raw_peso = data['Peso_Vagao'][i]
    if isinstance(raw_peso, str):
        peso_vagao = float(raw_peso.strip("%")) / 100
    else:
        peso_vagao = float(raw_peso)

    raw_mod = data['Modulo_Elasticidade'][i]
    if isinstance(raw_mod, str):
        modulo_elasticidade = float(raw_mod.strip("%")) / 1000
    else:
        modulo_elasticidade = float(raw_mod)

    irregularidade = int(data['Irregularidade'][i].lstrip("irreg"))
    
    dano_percentual = data["Dano_Percentual"][i] / 100

    matriz_x = np.array(data['Accel_X'][i])
    
    if all(x is None for x in matriz_x.flat) is True:
        continue
    
    matriz_y = np.array(data['Accel_Y'][i])
    matriz_z = np.array(data['Accel_Z'][i])
    
    data["ResultNodes"] = list(pd.Series(data["ResultNodes"]).ffill())
    node_ids = np.array(data['ResultNodes'][i]).flatten().astype(np.uint16)

    # Construção do DataFrame
    num_timesteps = matriz_x.shape[0]
    num_nodes = matriz_x.shape[1]
    vetor_tempo = np.arange(num_timesteps) * DT

    dados = {
        'ExperimentID': np.repeat(i, num_timesteps * num_nodes).astype(np.uint16),
        'Velocidade': np.repeat(velocidade, num_timesteps * num_nodes).astype(np.uint8),
        'Peso_Vagao': np.repeat(peso_vagao, num_timesteps * num_nodes).astype(np.float32),
        'Modulo_Elasticidade': np.repeat(modulo_elasticidade, num_timesteps * num_nodes).astype(np.float32),
        'Irregularidade': np.repeat(irregularidade, num_timesteps * num_nodes).astype(np.uint8),
        'Dano_Percentual': np.repeat(dano_percentual, num_timesteps * num_nodes).astype(np.float32),
        'NodeID': np.repeat(node_ids, num_timesteps).astype(np.uint16),
        'Time': np.tile(vetor_tempo, num_nodes).astype(np.float32),
        'Accel_X': matriz_x.ravel(order='F').astype(np.float64),
        'Accel_Y': matriz_y.ravel(order='F').astype(np.float64),
        'Accel_Z': matriz_z.ravel(order='F').astype(np.float64),
    }
    
    lista_de_dfs.append(pd.DataFrame(dados))

print("Concatenando e salvando...")
master_df = pd.concat(lista_de_dfs, ignore_index=True)

print(master_df)
print(master_df.info())

nome_saida = FILENAME.rsplit('.', 1)[0] + ".parquet"
master_df.to_parquet(nome_saida, index=False)
print(f"Sucesso! Arquivo salvo: {nome_saida}")