"""Pipeline de Autoencoder Convolucional 1D Esparso para detecção de defeitos em pontes."""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import welch
from tensorflow import keras
from tqdm import tqdm

from pipeline_template import BridgeDefectPipeline

RESIDUAL_BANDS = [(0, 1), (1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 32), (32, 64), (64, 128)]


def residual_spectrum(
    X: np.ndarray, X_hat: np.ndarray, latent: np.ndarray, fs: float, l1_activity: float
) -> dict[str, float]:
    """Descreve, por faixa de frequência, o resíduo de reconstrução e o espaço latente.

    Diagnóstico só com saudáveis: não ajusta nada no modelo nem no limiar. Uma faixa é
    considerada sem entrada quando tem menos de 0,1% da energia de entrada; nela, o
    resíduo é energia criada pelo próprio modelo.

    Args:
        X (np.ndarray): Sinais de entrada (amostras, tempo, canais).
        X_hat (np.ndarray): Reconstrução de ``X``, mesmo formato.
        latent (np.ndarray): Ativações do latente para ``X`` (amostras, tempo, canais).
        fs (float): Frequência de amostragem em Hz.
        l1_activity (float): Coeficiente L1 de atividade usado no treino.

    Returns:
        dict[str, float]: 'rel_error' (energia do resíduo ÷ energia de entrada),
            'created_out_of_band' (resíduo nas faixas sem entrada ÷ energia de entrada),
            por faixa 'in_share_{lo}-{hi}', 'res_share_{lo}-{hi}', 'not_rec_{lo}-{hi}' e
            'out_in_{lo}-{hi}' (NaN nas faixas sem entrada), e do latente 'latent_zeros',
            'active_channels', 'latent_channels' e 'l1_over_mse'.
    """
    residual = X - X_hat
    f, p_in = welch(X, fs=fs, nperseg=512, axis=1)
    _, p_res = welch(residual, fs=fs, nperseg=512, axis=1)
    _, p_out = welch(X_hat, fs=fs, nperseg=512, axis=1)
    p_in, p_res, p_out = (p.mean(axis=(0, 2)) for p in (p_in, p_res, p_out))
    total_in, total_res = p_in.sum(), p_res.sum()

    stats = {"rel_error": float(total_res / total_in), "created_out_of_band": 0.0}
    for lo, hi in RESIDUAL_BANDS:
        m = (f > lo) & (f <= hi) if lo else (f <= hi)
        band_in, band_res = p_in[m].sum(), p_res[m].sum()
        has_input = band_in / total_in > 1e-3
        stats[f"in_share_{lo}-{hi}"] = float(band_in / total_in)
        stats[f"res_share_{lo}-{hi}"] = float(band_res / total_res)
        # Sem entrada, as razões explodem e não têm significado.
        stats[f"not_rec_{lo}-{hi}"] = float(band_res / band_in) if has_input else np.nan
        stats[f"out_in_{lo}-{hi}"] = float(p_out[m].sum() / band_in) if has_input else np.nan
        if not has_input:
            stats["created_out_of_band"] += float(band_res / total_in)

    mse = float(np.mean(residual ** 2))
    stats["latent_zeros"] = float(np.mean(latent == 0))
    stats["active_channels"] = int((np.mean(latent > 0, axis=(0, 1)) > 0.01).sum())
    stats["latent_channels"] = int(latent.shape[2])
    stats["l1_over_mse"] = float(l1_activity * np.abs(latent).sum(axis=(1, 2)).mean() / mse)
    return stats


def compute_threshold(errors: np.ndarray, rule: str, k: float, percentile: float | None) -> float:
    """Calcula o limiar de anomalia a partir dos erros de reconstrução de saudáveis.

    Args:
        errors (np.ndarray): Erros de reconstrução das amostras saudáveis de treino.
        rule (str): "mean_std" (média + k * desvio) ou "percentile" (percentil dos erros).
        k (float): Número de desvios-padrão para a regra "mean_std".
        percentile (float | None): Percentil (0-100) para a regra "percentile".

    Returns:
        float: Limiar acima do qual uma amostra é classificada como anômala.

    Raises:
        ValueError: Se a regra for desconhecida ou o percentil estiver ausente/fora de 0-100.
    """
    if rule == "mean_std":
        return float(errors.mean() + k * errors.std())
    if rule == "percentile":
        if percentile is None or not 0 < percentile < 100:
            raise ValueError(f"threshold_percentile must be in (0, 100), got {percentile}")
        # Garante a fração (100 - percentil)% dos saudáveis acima do limiar, qualquer que
        # seja o formato da distribuição dos erros.
        return float(np.percentile(errors, percentile))
    raise ValueError(f"Unknown threshold_rule: {rule!r}")


class ThresholdedAutoencoder:
    """Autoencoder treinado acompanhado do limiar de erro de reconstrução.

    Expõe ``predict`` retornando rótulos binários, que é a interface esperada
    por ``BridgeDefectPipeline.run``.

    Args:
        model (keras.Model): Autoencoder treinado.
        threshold (float): Limiar de erro de reconstrução acima do qual a amostra é anômala.
        error_log (list[np.ndarray] | None): Lista onde ``predict`` acumula os erros de
            cada chamada, para diagnóstico posterior. None desativa o registro.
    """

    def __init__(
        self, model: keras.Model, threshold: float, error_log: list[np.ndarray] | None = None
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.error_log = error_log

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Calcula o erro quadrático médio (MSE) de reconstrução por amostra.

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
        errors = self.reconstruction_error(X)
        if self.error_log is not None:
            self.error_log.append(errors)
        return (errors > self.threshold).astype(int)


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
        # Configs anteriores ao E005 não têm estas chaves e usam média + k * desvio.
        self.threshold_rule = cfg.get("threshold_rule", "mean_std")
        self.threshold_percentile = cfg.get("threshold_percentile")
        self.random_state = random_state
        self.n_channels: int | None = None
        # Diagnóstico: metadados por experimento e erros de reconstrução por fold.
        self.sample_metadata: pd.DataFrame | None = None
        self.train_errors: list[np.ndarray] = []
        self.test_errors: list[np.ndarray] = []
        self.thresholds: list[float] = []
        self.residual_stats: list[dict[str, float]] = []

    def extract_features(self, raw_data: pd.DataFrame) -> pd.DataFrame: #TODO: adicionar typing @override
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

        meta_cols = ["ExperimentID", "Velocidade", "Peso_Vagao", "Modulo_Elasticidade",
                     "Irregularidade", "Dano_Percentual", "Class"]
        signals, classes, group_ids, metadata = [], [], [], []
        for _, exp_group in tqdm(data.groupby("ExperimentID"), desc="Building signal matrix"):
            channels = [
                # Filtra o registro completo e só depois recorta, evitando o transiente
                # de borda do filtfilt no ponto de corte.
                self._apply_filter(node_data[col].to_numpy())[:length]
                for _, node_data in exp_group.groupby("NodeID")
                for col in axes_cols
            ]
            signals.append(np.stack(channels, axis=-1).astype(np.float32).ravel())
            classes.append(exp_group["Class"].iloc[0])
            group_ids.append(self._compute_group_id(exp_group))
            metadata.append(exp_group[meta_cols].iloc[[0]])

        # Mesma ordem de linhas da matriz de sinais (usada no diagnóstico de erros).
        self.sample_metadata = pd.concat(metadata, ignore_index=True)
        self.sample_metadata["Group_ID"] = group_ids

        columns = [f"{c.split('_')[1]}_N{n}_t{t}" for t in range(length) for n in node_ids for c in axes_cols]
        feature_df = pd.DataFrame(np.vstack(signals), columns=columns)
        feature_df["Class"] = classes
        feature_df["Group_ID"] = group_ids
        return feature_df

    def select_features( #TODO: adicionar typing @override
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
        forma de onda; as acelerações brutas são pequenas (RMS ~5e-2), o que
        dificultaria a otimização da rede.

        Args:
            X_train (np.ndarray): Sinais de treino achatados (amostras, tempo*canais).
            X_test (np.nwdarray): Sinais de teste achatados (amostras, tempo*canais).

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

        O limiar vem só do erro de reconstrução das amostras saudáveis de treino, pela
        regra configurada em ``threshold_rule`` (ver ``compute_threshold``).

        Args:
            X_train (np.ndarray): Sinais de treino padronizados (amostras, tempo, canais).
            y_train (np.ndarray): Rótulos de treino (Classes 0 e 1).

        Returns:
            ThresholdedAutoencoder: Modelo treinado com o limiar calculado.
        """
        keras.utils.set_random_seed(self.random_state)
        # Na GPU, a semente não basta: as convoluções do cuDNN variam entre execuções.
        # Só na GPU, para não alterar os resultados dos runs em CPU já registrados.
        if tf.config.list_physical_devices("GPU"):
            tf.config.experimental.enable_op_determinism()
        X_healthy = X_train[y_train == 0]
        # Embaralha antes do fit: o validation_split do Keras pega as últimas amostras,
        # que estão ordenadas por arquivo/experimento.
        X_shuffled = np.random.default_rng(self.random_state).permutation(X_healthy)
        print(f"Training autoencoder on {X_healthy.shape[0]} healthy samples, input shape {X_healthy.shape[1:]}")

        model = self.build_model(*X_healthy.shape[1:])
        model.fit(
            X_shuffled, X_shuffled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[keras.callbacks.EarlyStopping(
                patience=self.early_stopping_patience, restore_best_weights=True
            )],
            verbose=2,
        )

        # Erros na ordem original (não embaralhada), para casar com os metadados.
        # Mesmo cálculo de ThresholdedAutoencoder.reconstruction_error, reaproveitando X_hat.
        X_hat = model.predict(X_healthy, verbose=0)
        errors = np.mean((X_healthy - X_hat) ** 2, axis=(1, 2))
        threshold = compute_threshold(errors, self.threshold_rule, self.threshold_k, self.threshold_percentile)
        self.train_errors.append(errors)
        self.thresholds.append(threshold)
        print(f"Train error: mean={errors.mean():.4f}, std={errors.std():.4f} -> threshold={threshold:.4f}")

        latent = keras.Model(model.input, model.get_layer("sparse_latent").output).predict(X_healthy, verbose=0)
        self.residual_stats.append(residual_spectrum(X_healthy, X_hat, latent, self.fs, self.l1_activity))
        return ThresholdedAutoencoder(model, threshold=threshold, error_log=self.test_errors)
