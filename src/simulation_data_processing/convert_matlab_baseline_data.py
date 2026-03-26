import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm # Para a barra de progresso

FILENAME = 'data/ponte_integra_com_defeitos.mat'
FREQUENCIA_HZ = 256
DT = 1.0 / FREQUENCIA_HZ

print(f"Iniciando o processamento do arquivo: {FILENAME}")

try:
    mat = scipy.io.loadmat(FILENAME)
    variavel_alvo = list(mat.keys())[-1]
    data = mat[variavel_alvo] # Shape (90, 1)
    
    # Lista para guardar os DataFrames de cada experimento
    lista_de_dfs = []
    
    num_experimentos = data.shape[0] if variavel_alvo == "Baseline" else data.shape[1]
    
    # tqdm envolve o loop para mostrar uma barra de progresso
    for i in tqdm(range(num_experimentos), desc="Processando experimentos"):
        
        # --- 1. Extrair o 'struct' do experimento atual ---
        # data[i][0] acessa o registro (struct) do experimento 'i'
        experimento = data[i][0] if variavel_alvo == "Baseline" else data[0][i]
        
        # --- 2. Extrair Metadados (que são escalares) ---
        # Usamos [0][0] para extrair o valor de dentro do array 1x1
        velocidade = experimento['Velocidade'][0][0]
        peso_vagao = float(experimento['Peso_Vagao'][0].strip("%")) / 100
        modulo_elasticidade = float(experimento['Modulo_Elasticidade'][0].strip("%")) / 1000
        irregularidade = int(experimento['Irregularidade'][0].lstrip("irreg"))

        # --- 3. Extrair Matrizes de Dados e Nós ---
        matriz_accel_x = experimento['Accel_X'] # Shape (N, 18)
        matriz_accel_y = experimento['Accel_Y'] # Shape (N, 18)
        matriz_accel_z = experimento['Accel_Z'] # Shape (N, 18)
        
        # .ravel() "achata" o array (18, 1) para (18,)
        node_ids = experimento['ResultNodes'].ravel() # Shape (18,)

        # --- 4. Obter dimensões e criar vetor de Tempo ---
        num_timesteps = matriz_accel_x.shape[0] # N (ex: 2233 ou 2010)
        num_nodes = matriz_accel_x.shape[1]     # K (sempre 18)
        
        # Cria o vetor de tempo para este experimento: [0, 0.0039, 0.0078, ...]
        vetor_tempo = np.arange(num_timesteps) * DT

        # --- 5. Reestruturar os dados (Mágica do NumPy) ---
        # Esta é a parte crucial. Convertemos as matrizes (N, K) para vetores (N*K)
        # de forma eficiente, sem usar loops 'for' lentos.
        
        # O (order='F') é essencial, pois "achata" a matriz por colunas
        # (todos os dados do nó 1, depois todos do nó 2, etc.)
        dados_longos = {
            'ExperimentID': np.repeat(i, num_timesteps * num_nodes).astype(np.uint16),
            'Velocidade': np.repeat(velocidade, num_timesteps * num_nodes).astype(np.uint8),
            'Peso_Vagao': np.repeat(peso_vagao, num_timesteps * num_nodes).astype(np.float32),
            'Modulo_Elasticidade': np.repeat(modulo_elasticidade, num_timesteps * num_nodes).astype(np.float32),
            'Irregularidade': np.repeat(irregularidade, num_timesteps * num_nodes).astype(np.uint8),
            
            # Repete cada ID de nó N vezes: [A, A, A, ..., B, B, B, ...]
            'NodeID': np.repeat(node_ids, num_timesteps).astype(np.uint16),
            
            # Repete o vetor de tempo K vezes: [t1, t2, t3, ..., t1, t2, t3, ...]
            'Time': np.tile(vetor_tempo, num_nodes).astype(np.float32),
            
            # Achata as matrizes de aceleração (ordem 'F')
            'Accel_X': matriz_accel_x.ravel(order='F').astype(np.float64),
            'Accel_Y': matriz_accel_y.ravel(order='F').astype(np.float64),
            'Accel_Z': matriz_accel_z.ravel(order='F').astype(np.float64),
        }

        # --- 6. Criar DF do experimento e adicionar à lista ---
        df_experimento = pd.DataFrame(dados_longos)
        lista_de_dfs.append(df_experimento)

    # --- 7. Concatenar tudo em um DataFrame Mestre ---
    print("\nConcatenando todos os experimentos...")
    master_df = pd.concat(lista_de_dfs, ignore_index=True)
    
    # Opcional: Reordenar colunas para melhor leitura
    colunas_ordenadas = [
        'ExperimentID', 'Velocidade', 'Peso_Vagao', 'Modulo_Elasticidade', 'Irregularidade',
        'NodeID', 'Time', 'Accel_X', 'Accel_Y', 'Accel_Z'
    ]
    master_df = master_df[colunas_ordenadas]

    print("\nProcessamento Concluído!")
    
    # --- 8. Mostrar resultado ---
    print("\nInformações do DataFrame final:")
    master_df.info()
    
    print("\n\nAmostra do DataFrame (primeiras 5 linhas):")
    print(master_df.head().to_string())
    
    # --- 9. Salvar o DataFrame em um arquivo CSV ou PARQUET ---
    # print("\nSalvando o DataFrame em um arquivo CSV...")
    # master_df.to_csv(FILENAME.rstrip("mat") + "csv", index=False)
    print("\nSalvando o DataFrame em um arquivo PARQUET...")
    master_df.to_parquet(FILENAME.rstrip("mat") + "parquet", index=False)


except FileNotFoundError:
    print(f"Erro: Arquivo '{FILENAME}' não encontrado.")
except KeyError as e:
    print(f"Erro: Não foi possível encontrar a variável '{e}' no arquivo .mat.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")