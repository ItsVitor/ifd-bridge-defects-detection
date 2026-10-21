"""Diagnóstico só com saudáveis: espectro do resíduo de reconstrução de um experimento.

Retreina o autoencoder do fold 1 (saudáveis de treino apenas, sem nenhum dado de teste)
com o ``config.yaml`` do experimento e mede, por faixa de frequência, quanto da energia
de entrada não é reconstruída e quanto o decoder cria fora da banda. Mede também a
esparsidade do espaço latente e o peso da penalidade L1 na perda.

Uso:
    python src/residual_spectrum.py outputs/autoencoders/E004_slug
"""

import argparse
import os

import numpy as np
from scipy.signal import welch
from tensorflow import keras

from autoencoders_pipeline import AutoencoderPipeline

BANDS = [(0, 1), (1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 32), (32, 64), (64, 128)]
FOLD_1_GROUPS = [1, 2, 3]


def residual_report(experiment_dir: str) -> str:
    """Treina o fold 1 só com saudáveis e descreve o resíduo por faixa de frequência.

    Args:
        experiment_dir (str): Pasta do experimento, contendo o ``config.yaml``.

    Returns:
        str: Relatório em texto, pronto para imprimir ou salvar.
    """
    pipeline = AutoencoderPipeline(
        os.path.join(experiment_dir, "config.yaml"), "./data",
        [(os.path.join("v2", "v2_healthy.parquet"), 0), (os.path.join("v2", "v2_slightly_damaged.parquet"), 0)],
    )
    df = pipeline.extract_features(pipeline.load_data())
    train = df[~df["Group_ID"].isin(FOLD_1_GROUPS)]
    X = train.drop(columns=["Class", "Group_ID"]).to_numpy()
    X_train, _ = pipeline.normalize(X, X[:1])
    model = pipeline.train_model(X_train, np.zeros(len(X_train), dtype=int)).model

    X_hat = model.predict(X_train, verbose=0)
    f, p_in = welch(X_train, fs=pipeline.fs, nperseg=512, axis=1)
    _, p_res = welch(X_train - X_hat, fs=pipeline.fs, nperseg=512, axis=1)
    _, p_out = welch(X_hat, fs=pipeline.fs, nperseg=512, axis=1)
    p_in, p_res, p_out = (p.mean(axis=(0, 2)) for p in (p_in, p_res, p_out))

    lines = [
        f"Experiment: {experiment_dir} | fold 1, healthy training samples only ({len(X_train)})",
        f"Threshold (fold 1): {pipeline.thresholds[0]:.4f}",
        f"Relative error (residual energy / input energy): {p_res.sum() / p_in.sum():.4f}",
        "",
        "band (Hz) | input share | residual share | band not reconstructed | output/input energy",
    ]
    for lo, hi in BANDS:
        m = (f > lo) & (f <= hi) if lo else (f <= hi)
        band_in = p_in[m].sum()
        # Fora da banda do filtro a entrada é ~0: as razões explodem e não têm significado.
        ratio = f"{p_res[m].sum() / band_in:.3f}" if band_in / p_in.sum() > 1e-3 else "n/a (no input)"
        out = f"{p_out[m].sum() / band_in:.3f}" if band_in / p_in.sum() > 1e-3 else "n/a"
        lines.append(f"{lo:>3}-{hi:<3}    |   {band_in / p_in.sum():.4f}    |    {p_res[m].sum() / p_res.sum():.4f}"
                     f"      |   {ratio:<20} |   {out}")

    latent = keras.Model(model.input, model.get_layer("sparse_latent").output).predict(X_train, verbose=0)
    mse = float(np.mean((X_train - X_hat) ** 2))
    active = int((np.mean(latent > 0, axis=(0, 1)) > 0.01).sum())
    lines += [
        "",
        f"Latent shape {latent.shape[1:]} ({int(np.prod(latent.shape[1:]))} values), "
        f"zeros: {np.mean(latent == 0):.3f}, active channels: {active}/{latent.shape[2]}",
        f"L1 penalty / MSE: {pipeline.l1_activity * np.abs(latent).sum(axis=(1, 2)).mean() / mse:.4f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Espectro do resíduo de reconstrução (fold 1, só saudáveis).")
    parser.add_argument("experiment_dir", help="Pasta do experimento, contendo o config.yaml.")
    experiment_dir = parser.parse_args().experiment_dir
    report = residual_report(experiment_dir)
    print("\n" + report)
    with open(os.path.join(experiment_dir, "residual_spectrum.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
