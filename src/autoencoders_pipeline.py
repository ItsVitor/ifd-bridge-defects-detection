"""Pipeline de Autoencoder Convolucional 1D Esparso para detecção de defeitos em pontes."""

import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm

from pipeline_template import BridgeDefectPipeline


class ThresholdedAutoencoder:
    """Autoencoder treinado acompanhado do limiar de erro de reconstrução.

    Expõe ``predict`` retornando rótulos binários, que é a interface esperada
    por ``BridgeDefectPipeline.run``.

    Args:
        model (keras.Model): Autoencoder treinado.
        threshold (float): Limiar de erro de reconstrução acima do qual a amostra é anômala.
    """

    def __init__(self, model: keras.Model, threshold: float) -> None:
        self.model = model
        self.threshold = threshold

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Calcula o erro quadrático médio de reconstrução por amostra.

        Args:
            X (np.ndarray): Sinais no formato (amostras, tempo, canais).

        Returns:
            np.ndarray: Vetor (amostras,) com o MSE de cada amostra.
        """
        X_hat = self.model.predict(X, verbose=0)
        return np.mean((X - X_hat) ** 2, axis=(1, 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classifica amostras comparando o erro de reconstrução com o limiar.

        Args:
            X (np.ndarray): Sinais no formato (amostras, tempo, canais).

        Returns:
            np.ndarray: Rótulos binários (0=saudável, 1=danificado).
        """
        return (self.reconstruction_error(X) > self.threshold).astype(int)


class AutoencoderPipeline(BridgeDefectPipeline):
    """Pipeline com Autoencoder Conv1D Esparso, treinado apenas com amostras saudáveis.

    Em vez de features estatísticas, consome o sinal bruto de aceleração de cada
    experimento como uma série temporal multicanal (canal = nó x eixo).

    Args:
        config_path (str): Caminho para o arquivo YAML de configuração.
        data_dir (str): Caminho para o diretório de dados.
        file_config (list[tuple[str, int]] | None): Lista de (nome_arquivo, rótulo_classe).
        random_state (int): Semente para reprodutibilidade.
    """

    def __init__(
        self,
        config_path: str,
        data_dir: str = "./data",
        file_config: list[tuple[str, int]] | None = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(config_path, data_dir, file_config)
        cfg = self._load_config(config_path)["model"]
        self.encoder_filters = cfg["encoder_filters"]
        self.pool_sizes = cfg["pool_sizes"]
        self.kernel_size = cfg["kernel_size"]
        self.l1_activity = cfg["l1_activity"]
        self.learning_rate = cfg["learning_rate"]
        self.epochs = cfg["epochs"]
        self.batch_size = cfg["batch_size"]
        self.validation_split = cfg["validation_split"]
        self.early_stopping_patience = cfg["early_stopping_patience"]
        self.threshold_k = cfg["threshold_k"]
        self.random_state = random_state
        self.n_channels: int | None = None

    def extract_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Monta a matriz de sinais brutos, uma linha por experimento.

        Cada linha contém o sinal (tempo x canais) achatado; ``normalize`` o
        remodela para (amostras, tempo, canais) antes do treino.

        Args:
            raw_data (pd.DataFrame): Dados brutos de aceleração vindos de ``load_data``.

        Returns:
            pd.DataFrame: Matriz com colunas ``{eixo}_N{nó}_t{i}`` mais 'Class' e 'Group_ID'.
        """
        node_ids = sorted(raw_data["NodeID"].unique())
        if self.nodes_to_use is not None:
            node_ids = [n for n in node_ids if n in self.nodes_to_use]
        axes_cols = [f"Accel_{a}" for a in self.axes_to_use]
        self.n_channels = len(node_ids) * len(axes_cols)

        data = raw_data[raw_data["NodeID"].isin(node_ids)].sort_values(
            ["ExperimentID", "NodeID", "Time"]
        )

        # A duração do registro varia com a velocidade do trem (1827/2010/2233 amostras).
        # ponytail: recorta todos os sinais no menor comprimento (sem reamostrar, pois
        # isso deslocaria as frequências modais). Perde-se a cauda de vibração livre das
        # passagens mais lentas; alternativa futura: zero-padding com erro mascarado.
        down_factor = int(np.prod(self.pool_sizes))
        min_length = data.groupby(["ExperimentID", "NodeID"]).size().min()
        length = (min_length // down_factor) * down_factor
        print(f"Signal length: {min_length} -> cropped to {length} samples ({length / self.fs:.2f} s)")

        signals, classes, group_ids = [], [], []
        for _, exp_group in tqdm(data.groupby("ExperimentID"), desc="Building signal matrix"):
            channels = [
                self._apply_filter(node_data[col].to_numpy()[:length])
                for _, node_data in exp_group.groupby("NodeID")
                for col in axes_cols
            ]
            signals.append(np.stack(channels, axis=-1).astype(np.float32).ravel())
            classes.append(exp_group["Class"].iloc[0])
            group_ids.append(self._compute_group_id(exp_group))

        columns = [f"{c.split('_')[1]}_N{n}_t{t}" for t in range(length) for n in node_ids for c in axes_cols]
        feature_df = pd.DataFrame(np.vstack(signals), columns=columns)
        feature_df["Class"] = classes
        feature_df["Group_ID"] = group_ids
        return feature_df

    def select_features(
        self, X_train: np.ndarray, X_test: np.ndarray, fold_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Não aplica filtros de variância/correlação: as colunas são instantes de tempo.

        Remover colunas quebraria a continuidade temporal que a convolução explora.

        Args:
            X_train (np.ndarray): Sinais de treino achatados.
            X_test (np.ndarray): Sinais de teste achatados.
            fold_idx (int): Número do fold atual.

        Returns:
            tuple[np.ndarray, np.ndarray]: X_train e X_test inalterados.
        """
        return X_train, X_test

    def normalize(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Padroniza cada canal (média 0, desvio 1) com estatísticas apenas do treino.

        A padronização é por canal, e não por instante de tempo, para preservar a
        forma de onda; as acelerações brutas têm ordem de 1e-3, o que dificultaria
        a otimização da rede.

        Args:
            X_train (np.ndarray): Sinais de treino achatados (amostras, tempo*canais).
            X_test (np.ndarray): Sinais de teste achatados (amostras, tempo*canais).

        Returns:
            tuple[np.ndarray, np.ndarray]: X_train e X_test no formato (amostras, tempo, canais).
        """
        X_train = X_train.reshape(len(X_train), -1, self.n_channels).astype(np.float32)
        X_test = X_test.reshape(len(X_test), -1, self.n_channels).astype(np.float32)
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-12
        return (X_train - mean) / std, (X_test - mean) / std

    def build_model(self, n_timesteps: int, n_channels: int) -> keras.Model:
        """Constrói o autoencoder Conv1D esparso (encoder e decoder espelhados).

        Args:
            n_timesteps (int): Número de amostras temporais por sinal.
            n_channels (int): Número de canais (nós x eixos).

        Returns:
            keras.Model: Autoencoder compilado com Adam e perda MSE.
        """
        inputs = keras.Input(shape=(n_timesteps, n_channels))
        x = inputs
        for filters, pool in zip(self.encoder_filters, self.pool_sizes):
            x = keras.layers.Conv1D(filters, self.kernel_size, padding="same", activation="relu")(x)
            x = keras.layers.MaxPooling1D(pool)(x)
        # Gargalo esparso: a penalidade L1 sobre as ativações força a maioria a zero.
        x = keras.layers.ActivityRegularization(l1=self.l1_activity, name="sparse_latent")(x)
        for filters, pool in zip(reversed(self.encoder_filters), reversed(self.pool_sizes)):
            x = keras.layers.Conv1D(filters, self.kernel_size, padding="same", activation="relu")(x)
            x = keras.layers.UpSampling1D(pool)(x)
        # Saída linear: o sinal padronizado é contínuo e assume valores negativos.
        outputs = keras.layers.Conv1D(n_channels, self.kernel_size, padding="same")(x)

        model = keras.Model(inputs, outputs, name="sparse_conv1d_autoencoder")
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")
        return model

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> ThresholdedAutoencoder:
        """Treina o autoencoder com amostras saudáveis e define o limiar de anomalia.

        Limiar = média + k * desvio-padrão do erro de reconstrução das amostras
        saudáveis de treino.

        Args:
            X_train (np.ndarray): Sinais de treino padronizados (amostras, tempo, canais).
            y_train (np.ndarray): Rótulos de treino (Classes 0 e 1).

        Returns:
            ThresholdedAutoencoder: Modelo treinado com o limiar calculado.
        """
        keras.utils.set_random_seed(self.random_state)
        # Embaralha antes do fit: o validation_split do Keras pega as últimas amostras,
        # que estão ordenadas por arquivo/experimento.
        X_healthy = np.random.default_rng(self.random_state).permutation(X_train[y_train == 0])
        print(f"Training autoencoder on {X_healthy.shape[0]} healthy samples, input shape {X_healthy.shape[1:]}")

        model = self.build_model(*X_healthy.shape[1:])
        model.fit(
            X_healthy, X_healthy,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[keras.callbacks.EarlyStopping(
                patience=self.early_stopping_patience, restore_best_weights=True
            )],
            verbose=2,
        )

        detector = ThresholdedAutoencoder(model, threshold=0.0)
        errors = detector.reconstruction_error(X_healthy)
        detector.threshold = float(errors.mean() + self.threshold_k * errors.std())
        print(f"Train error: mean={errors.mean():.4f}, std={errors.std():.4f} -> threshold={detector.threshold:.4f}")
        return detector
