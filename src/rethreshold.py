"""Reavalia um experimento com outra regra de limiar, sem retreinar o autoencoder.

O erro de reconstrução de cada amostra não depende do limiar. Por isso, um experimento
que muda só a regra do limiar pode reaproveitar os erros salvos do experimento pai
(``test_errors.parquet`` e ``train_errors.parquet``): o resultado é idêntico ao de um
run completo, que reproduziria os mesmos erros (o treino é determinístico).

O limiar de cada fold continua vindo só dos erros dos saudáveis de treino daquele fold.

Uso:
    python src/rethreshold.py outputs/autoencoders/E004_pai outputs/autoencoders/E005_filho
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from autoencoders_pipeline import compute_threshold
from test_autoencoders import git_revision, print_error_report

THRESHOLD_KEYS = {"threshold_rule", "threshold_k", "threshold_percentile", "threshold_group"}


def load_config(experiment_dir: str) -> dict:
    """Lê o ``config.yaml`` de um experimento.

    Args:
        experiment_dir (str): Pasta do experimento.

    Returns:
        dict: Configuração do experimento.
    """
    with open(os.path.join(experiment_dir, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_only_threshold_changed(parent: dict, child: dict) -> None:
    """Garante que os dois configs diferem apenas nas chaves da regra do limiar.

    Args:
        parent (dict): Configuração do experimento pai.
        child (dict): Configuração do experimento filho.

    Raises:
        ValueError: Se algo além da regra do limiar tiver mudado.
    """
    strip = lambda cfg: {**cfg, "model": {k: v for k, v in cfg["model"].items() if k not in THRESHOLD_KEYS}}
    if strip(parent) != strip(child):
        raise ValueError("Configs differ beyond the threshold rule; a full run is required.")


def fold_metrics(test_df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Calcula acurácia e recall por fold, como em ``BridgeDefectPipeline.evaluate_fold``.

    Args:
        test_df (pd.DataFrame): Amostras de teste com 'fold', 'Class', 'error' e 'threshold'.

    Returns:
        tuple[list[float], list[float]]: Acurácias e recalls de cada fold, em ordem.
    """
    accuracies, recalls = [], []
    for _, g in test_df.groupby("fold"):
        y_true, y_pred = g["Class"].to_numpy(), (g["error"] > g["threshold"]).to_numpy().astype(int)
        accuracies.append(float((y_true == y_pred).mean()))
        positives = y_true == 1
        recalls.append(float(y_pred[positives].mean()) if positives.any() else 0.0)
    return accuracies, recalls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reavalia um experimento com outra regra de limiar.")
    parser.add_argument("parent_dir", help="Pasta do experimento pai, com os parquets de erro.")
    parser.add_argument("experiment_dir", help="Pasta do novo experimento, com o config.yaml.")
    args = parser.parse_args()

    parent_cfg, child_cfg = load_config(args.parent_dir), load_config(args.experiment_dir)
    check_only_threshold_changed(parent_cfg, child_cfg)
    model_cfg = child_cfg["model"]
    rule = model_cfg.get("threshold_rule", "mean_std")
    k, percentile = model_cfg["threshold_k"], model_cfg.get("threshold_percentile")

    # threshold_group (opcional): coluna de metadado (ex.: "Velocidade") que separa os
    # saudáveis de treino de cada fold; cada grupo recebe o próprio limiar, e cada passagem
    # de teste usa o limiar do seu grupo. Supõe que o metadado é medido em operação.
    group = model_cfg.get("threshold_group")
    keys = ["fold"] + ([group] if group else [])

    test_df = pd.read_parquet(os.path.join(args.parent_dir, "test_errors.parquet"))
    train_df = pd.read_parquet(os.path.join(args.parent_dir, "train_errors.parquet"))
    thresholds = (
        train_df.groupby(keys)["error"]
        .agg(lambda e: compute_threshold(e.to_numpy(), rule, k, percentile))
        .rename("threshold").reset_index()
    )
    test_df = test_df.drop(columns="threshold").merge(thresholds, on=keys, how="left", validate="many_to_one")
    if test_df["threshold"].isna().any():
        raise ValueError("Some test groups have no healthy training samples to set a threshold.")
    test_df.to_parquet(os.path.join(args.experiment_dir, "test_errors.parquet"))
    train_df.to_parquet(os.path.join(args.experiment_dir, "train_errors.parquet"))

    print("=" * 60)
    print("SPARSE CONV1D AUTOENCODER: RE-THRESHOLD (NO RETRAINING)")
    print("=" * 60)
    print(f"\nExperiment: {args.experiment_dir}")
    print(f"Reconstruction errors from: {args.parent_dir}")
    print(f"Code revision: {git_revision()}")
    print(f"Threshold rule: {rule}, threshold_k={k}, threshold_percentile={percentile}, threshold_group={group}")
    merged = train_df.merge(thresholds, on=keys)
    summary = (merged.assign(above=merged["error"] > merged["threshold"])
               .groupby(keys).agg(threshold=("threshold", "first"), above=("above", "mean")).reset_index())
    for row in summary.to_dict("records"):
        print(f"  {', '.join(f'{c}={row[c]:g}' for c in keys)}: threshold={row['threshold']:.4f}  "
              f"healthy train above: {row['above']:.3f}")

    accuracies, recalls = fold_metrics(test_df)
    y_true = test_df["Class"].to_numpy()
    y_pred = (test_df["error"] > test_df["threshold"]).to_numpy().astype(int)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nMean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Accuracy: {np.std(accuracies):.4f}")
    print(f"\nMean Recall: {np.mean(recalls):.4f}")
    print(f"Std Recall: {np.std(recalls):.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {((y_true == 0) & (y_pred == 0)).sum()}  FP: {((y_true == 0) & (y_pred == 1)).sum()}")
    print(f"FN: {((y_true == 1) & (y_pred == 0)).sum()}  TP: {((y_true == 1) & (y_pred == 1)).sum()}")
    print("\nFold Accuracies:")
    for i, acc in enumerate(accuracies, 1):
        print(f"  Fold {i:2d}: {acc:.4f}")
    print("\nFold Recalls:")
    for i, recall in enumerate(recalls, 1):
        print(f"  Fold {i:2d}: {recall:.4f}")

    print_error_report(test_df, train_df, args.experiment_dir)
