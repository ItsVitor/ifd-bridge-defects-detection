"""Verificação rápida de ``residual_spectrum`` com sinais sintéticos de resposta conhecida.

Entrada: seno de 10 Hz. Reconstrução: 0,9 × entrada + seno de 50 Hz com amplitude 0,05.
Assim, 1% da energia de 8–12 Hz não é reconstruída, o modelo cria energia em 32–64 Hz
igual a 0,25% da entrada, e o resto do resultado segue destes dois valores.

Uso:
    python src/check_residual_spectrum.py
"""

import numpy as np

from autoencoders_pipeline import residual_spectrum

FS = 256
t = np.arange(1824) / FS
x = np.sin(2 * np.pi * 10 * t)
X = np.tile(x[None, :, None], (4, 1, 3))
X_hat = 0.9 * X + 0.05 * np.sin(2 * np.pi * 50 * t)[None, :, None]
# Latente com um canal morto (sempre zero) e outro sempre igual a 1.
latent = np.stack([np.zeros((4, 10)), np.ones((4, 10))], axis=-1)

s = residual_spectrum(X, X_hat, latent, FS, l1_activity=1e-3)
mse = 0.5 * 0.1 ** 2 + 0.5 * 0.05 ** 2
assert np.isclose(s["in_share_8-12"], 1, atol=1e-3), s["in_share_8-12"]
assert np.isclose(s["not_rec_8-12"], 0.01, rtol=0.02), s["not_rec_8-12"]
assert np.isclose(s["created_out_of_band"], 0.0025, rtol=0.02), s["created_out_of_band"]
assert np.isclose(s["rel_error"], 0.0125, rtol=0.02), s["rel_error"]
assert np.isnan(s["not_rec_16-20"]) and np.isnan(s["out_in_64-128"])
assert s["latent_zeros"] == 0.5 and s["active_channels"] == 1 and s["latent_channels"] == 2
# rtol: 7,125 s não é um número inteiro de períodos de 50 Hz, então a média de sen² não é exatamente 0,5.
assert np.isclose(s["l1_over_mse"], 1e-3 * 10 / mse, rtol=1e-3), s["l1_over_mse"]
print("residual_spectrum: OK")
