"""Verificação rápida dos diagnósticos e da entrada log-PSD com sinais sintéticos de resposta conhecida.

``residual_spectrum``: entrada seno de 10 Hz; reconstrução 0,9 × entrada + seno de 50 Hz com
amplitude 0,05. Assim, 1% da energia de 8–12 Hz não é reconstruída, o modelo cria energia em
32–64 Hz igual a 0,25% da entrada, e o resto do resultado segue destes dois valores.

``log_psd``: senos de 10 Hz e 5 Hz em dois canais devem gerar picos nessas frequências, com a
potência do seno (0,5) na faixa de 1–20,5 Hz.

``residual_by_frequency``: um erro unitário só em 4–8 Hz deve aparecer só nessa faixa.

Uso:
    python src/check_residual_spectrum.py
"""

import numpy as np

from autoencoders_pipeline import log_psd, residual_by_frequency, residual_spectrum

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

signal = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 5 * t)], axis=-1)
freqs, L = log_psd(signal, FS, nperseg=512, band=(1.0, 20.5))
assert len(freqs) == 40 and freqs[0] == 1.0 and freqs[-1] == 20.5, freqs
assert freqs[L[:, 0].argmax()] == 10.0 and freqs[L[:, 1].argmax()] == 5.0
# PSD em densidade: a soma × resolução (0,5 Hz) é a potência do seno.
assert np.allclose((10 ** L).sum(axis=0) * 0.5, 0.5, rtol=0.02), (10 ** L).sum(axis=0) * 0.5
print("log_psd: OK")

Xf = np.zeros((2, 40, 3))
Xf_hat = Xf.copy()
Xf_hat[:, (freqs > 4) & (freqs <= 8), :] = 1.0
s = residual_by_frequency(Xf, Xf_hat, latent, freqs, l1_activity=0.0)
assert s["mse_4-8"] == 1.0 and s["mse_1-4"] == 0.0 and s["mse_16-20"] == 0.0, s
assert {"mse_0-1", "mse_20-32"} <= s.keys() and "mse_32-64" not in s  # 1,0 Hz e 20,5 Hz nas bordas
assert s["active_channels"] == 1
print("residual_by_frequency: OK")
