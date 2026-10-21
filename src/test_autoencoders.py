"""Script de teste do pipeline de Autoencoder Conv1D Esparso."""

import argparse
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from autoencoders_pipeline import AutoencoderPipeline

EOV_FACTORS = ["Velocidade", "Peso_Vagao", "Modulo_Elasticidade", "Irregularidade"]


def git_revision() -> str:
    """Identifica a versão do código usada no run (commit atual e alterações pendentes em src/).

    Returns:
        str: Hash curto do commit, com aviso se houver alterações não commitadas em src/,
            ou "unknown" se o git não estiver disponível.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit} (WARNING: uncommitted changes in src/)" if dirty else commit


def eta_squared(values: pd.Series, labels: pd.Series) -> float:
    """Calcula a fração da variância de ``values`` explicada pelos níveis de ``labels`` (η²).

    Args:
        values (pd.Series): Valores numéricos (ex.: log do erro de reconstrução).
        labels (pd.Series): Nível do fator para cada valor.

    Returns:
        float: η² entre 0 e 1.
    """
    group_means = values.groupby(labels).transform("mean")
    return float(((group_means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum())


def severity_report(test_df: pd.DataFrame) -> pd.DataFrame:
    """Resume, por severidade, recall, FPR dos saudáveis e AUC (uso apenas na avaliação).

    O recall de uma severidade só tem significado comparado ao FPR: um detector que
    sorteia alarmes teria recall igual ao FPR. A AUC complementa sem depender do limiar.

    Args:
        test_df (pd.DataFrame): Amostras de teste com colunas 'Class', 'Dano_Percentual',
            'fold', 'error' e 'threshold'.

    Returns:
        pd.DataFrame: Uma linha por severidade com 'severity', 'n', 'recall', 'fpr',
            'recall_minus_fpr', 'auc_mean' e 'auc_std' (AUC calculada dentro de cada fold).
    """
    alarm = test_df["error"] > test_df["threshold"]
    healthy = test_df["Class"] == 0
    fpr = float(alarm[healthy].mean())
    rows = []
    for dmg in sorted(test_df.loc[~healthy, "Dano_Percentual"].unique()):
        is_dmg = test_df["Dano_Percentual"] == dmg
        subset = test_df[healthy | is_dmg]
        aucs = [roc_auc_score(g["Class"], g["error"]) for _, g in subset.groupby("fold")]
        recall = float(alarm[is_dmg].mean())
        rows.append({
            "severity": dmg, "n": int(is_dmg.sum()), "recall": recall, "fpr": fpr,
            "recall_minus_fpr": recall - fpr, "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
        })
    return pd.DataFrame(rows)


def plot_error_distribution(test_df: pd.DataFrame, path: str) -> None:
    """Salva a distribuição do erro de reconstrução por severidade (uso apenas na avaliação).

    O erro é dividido pelo limiar do próprio fold, de modo que 1 é a fronteira de decisão
    em todos os folds e as amostras de folds diferentes ficam comparáveis.

    Args:
        test_df (pd.DataFrame): Amostras de teste com colunas 'Class', 'Dano_Percentual',
            'error' e 'threshold'.
        path (str): Caminho do arquivo PNG de saída.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    ratio = test_df["error"] / test_df["threshold"]
    groups = [("healthy", ratio[test_df["Class"] == 0])] + [
        (f"{dmg:g}", ratio[test_df["Dano_Percentual"] == dmg])
        for dmg in sorted(test_df.loc[test_df["Class"] == 1, "Dano_Percentual"].unique())
    ]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(groups)))

    fig, (ax_box, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Boxplot com pontos sobrepostos: mostra mediana, dispersão e cauda de cada grupo.
    ax_box.boxplot([g for _, g in groups], showfliers=False, widths=0.5)
    rng = np.random.default_rng(0)
    for i, ((_, g), c) in enumerate(zip(groups, colors), 1):
        ax_box.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g, s=4, alpha=0.35, color=c)
    ax_box.set_xticks(range(1, len(groups) + 1), [name for name, _ in groups])
    ax_box.set_yscale("log")
    ax_box.axhline(1, color="red", ls="--", lw=1, label="fold threshold")
    ax_box.set_xlabel("Dano_Percentual")
    ax_box.set_ylabel("reconstruction error / fold threshold (log)")
    ax_box.legend()

    # ECDF: a fração acima de x=1 é o FPR (saudáveis) ou o recall (danificados).
    for (name, g), c in zip(groups, colors):
        values = np.sort(g.to_numpy())
        ax_cdf.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                    color=c, lw=2.5 if name == "healthy" else 1.5, label=name)
    ax_cdf.set_xscale("log")
    # Rótulos simples no eixo log: os automáticos se sobrepõem.
    ax_cdf.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax_cdf.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_cdf.set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7])
    ax_cdf.axvline(1, color="red", ls="--", lw=1)
    ax_cdf.set_xlabel("reconstruction error / fold threshold (log)")
    ax_cdf.set_ylabel("cumulative fraction of samples")
    ax_cdf.legend(title="Dano_Percentual")

    fig.suptitle("Reconstruction error distribution by damage severity (test samples, all folds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roda um experimento do autoencoder (ver docs/experiments.md).")
    parser.add_argument("experiment_dir", help="Pasta do experimento, contendo o config.yaml congelado.")
    experiment_dir = parser.parse_args().experiment_dir
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.isfile(config_path):
        parser.error(f"config.yaml not found in {experiment_dir}")

    pipeline = AutoencoderPipeline(
        config_path=config_path,
        data_dir=os.path.join(".", "data"),
        file_config=[
            (os.path.join("v2", "v2_healthy.parquet"), 0),
            (os.path.join("v2", "v2_slightly_damaged.parquet"), 0),
            (os.path.join("v2", "v2_damaged_d1.parquet"), 1),
            (os.path.join("v2", "v2_damaged_d2.parquet"), 1),
            (os.path.join("v2", "v2_damaged_d3.parquet"), 1),
            (os.path.join("v2", "v2_damaged_d4.parquet"), 1),
            (os.path.join("v2", "v2_damaged_d5.parquet"), 1)
        ],
    )

    print("=" * 60)
    print("SPARSE CONV1D AUTOENCODER PIPELINE TEST")
    print("=" * 60)
    print(f"\nExperiment: {experiment_dir}")
    print(f"Code revision: {git_revision()}")
    print(
        f"\nHyperparameters: filters={pipeline.encoder_filters}, pools={pipeline.pool_sizes}, "
        f"kernel={pipeline.kernel_size}, l1={pipeline.l1_activity}, lr={pipeline.learning_rate}, "
        f"threshold_rule={pipeline.threshold_rule}, threshold_k={pipeline.threshold_k}, "
        f"threshold_percentile={pipeline.threshold_percentile}"
    )

    print("\nRunning full pipeline...")
    results = pipeline.run()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nMean Accuracy: {results['mean_accuracy']:.4f}")
    print(f"Std Accuracy: {results['std_accuracy']:.4f}")
    print(f"\nMean Recall: {results['mean_recall']:.4f}")
    print(f"Std Recall: {results['std_recall']:.4f}")
    tn, fp, fn, tp = results["confusion_matrix"].ravel()
    print("\nConfusion Matrix:")
    print(f"TN: {tn}  FP: {fp}")
    print(f"FN: {fn}  TP: {tp}")
    print(f"\nFold Accuracies:")
    for i, acc in enumerate(results['fold_accuracies'], 1):
        print(f"  Fold {i:2d}: {acc:.4f}")
    print(f"\nFold Recalls:")
    for i, recall in enumerate(results['fold_recalls'], 1):
        print(f"  Fold {i:2d}: {recall:.4f}")

    preds = pipeline._cv_predictions

    # --- Diagnóstico: erro de reconstrução por amostra (uso apenas na avaliação) ---
    meta = pipeline.sample_metadata.copy()
    meta["Peso_Vagao"] = meta["Peso_Vagao"].round(2)
    meta["Modulo_Elasticidade"] = meta["Modulo_Elasticidade"].round(3)
    y = meta["Class"].to_numpy()
    groups = meta["Group_ID"].to_numpy()
    splits = list(pipeline._create_cv_splitter(groups).split(meta, y, groups))

    # Os erros de teste foram acumulados na mesma ordem dos folds do run().
    assert np.array_equal(preds["indices"], np.concatenate([te for _, te in splits]))

    test_df = meta.iloc[preds["indices"]].reset_index(drop=True)
    test_df["fold"] = np.concatenate([np.full(len(te), i) for i, (_, te) in enumerate(splits, 1)])
    test_df["error"] = np.concatenate(pipeline.test_errors)
    test_df["threshold"] = test_df["fold"].map(dict(enumerate(pipeline.thresholds, 1)))
    train_df = pd.concat(
        [meta.iloc[tr[y[tr] == 0]].assign(fold=i, error=err)
         for i, ((tr, _), err) in enumerate(zip(splits, pipeline.train_errors), 1)],
        ignore_index=True,
    )
    test_df.to_parquet(os.path.join(experiment_dir, "test_errors.parquet"))
    train_df.to_parquet(os.path.join(experiment_dir, "train_errors.parquet"))

    # As predições do run() são exatamente erro > limiar do fold.
    assert np.array_equal(preds["y_pred"], (test_df["error"] > test_df["threshold"]).astype(int))

    print("\n" + "=" * 60)
    print("PERFORMANCE BY DAMAGE SEVERITY")
    print("=" * 60)
    print("recall: damaged of that severity flagged | fpr: all healthy test samples flagged")
    print("auc: within-fold AUC (healthy test vs. that severity), mean and std over folds")
    print(severity_report(test_df).round(4).to_string(index=False))
    plot_path = os.path.join(experiment_dir, "error_distribution.png")
    plot_error_distribution(test_df, plot_path)
    print(f"\nError distribution plot saved to {plot_path}")

    print("\n" + "=" * 60)
    print("DIAGNOSTICS: RECONSTRUCTION ERROR")
    print("=" * 60)
    print(f"\nPooled AUC (all folds): {roc_auc_score(test_df['Class'], test_df['error']):.4f}")

    rows = []
    for fold, g in test_df.groupby("fold"):
        rows.append({
            "fold": fold,
            "peso": g["Peso_Vagao"].iloc[0],
            "modulo": g["Modulo_Elasticidade"].iloc[0],
            "train_healthy": train_df.loc[train_df["fold"] == fold, "error"].mean(),
            "test_healthy": g.loc[g["Class"] == 0, "error"].mean(),
            "test_damaged": g.loc[g["Class"] == 1, "error"].mean(),
            "threshold": g["threshold"].iloc[0],
            "auc": roc_auc_score(g["Class"], g["error"]),
        })
    print("\nPer fold (mean reconstruction error and within-fold AUC):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nHealthy TRAIN error vs. EOV (eta^2 of log-error, mean over folds):")
    for factor in EOV_FACTORS:
        etas = [eta_squared(np.log(g["error"]), g[factor]) for _, g in train_df.groupby("fold")]
        level_means = {k: round(float(v), 4) for k, v in train_df.groupby(factor)["error"].mean().items()}
        print(f"  {factor:20s} eta^2={np.mean(etas):.3f}  mean error by level: {level_means}")

    healthy_test = test_df[test_df["Class"] == 0]
    print("\nHealthy TEST error by Irregularidade:",
          {k: round(float(v), 4) for k, v in healthy_test.groupby("Irregularidade")["error"].mean().items()})
