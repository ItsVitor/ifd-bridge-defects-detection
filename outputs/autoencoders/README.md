# Experimentos: Autoencoder Conv1D Esparso

Fluxo e regras: `docs/experiments.md`. Modelo de registro: `docs/experiment_template.md`.

- **Base atual:** E003.
- **Variantes avaliadas nos folds de teste até aqui:** 3 (E001–E003).

| ID | Título | Pai | Mudança | Hipótese (resumo) | Resultado principal | Decisão | Status |
|---|---|---|---|---|---|---|---|
| [E001](E001_baseline-tempo/README.md) | baseline-tempo | — | Sinal no tempo, sem filtro, 100 épocas | AE treinado em saudáveis reconstrói pior passagens com dano | AUC ≈ 0,5 até sev. 0,5; 0,90 em 1,0; FPR 17% | Base inicial | Concluído (retroativo) |
| [E002](E002_200-epocas/README.md) | 200-epocas | E001 | `epochs` 100 → 200 | Mais épocas reduzem o erro dos saudáveis e o modelo converge | `val_loss` 0,330 → 0,312; AUC por severidade inalterada | Adotada (`val_loss`) | Concluído (retroativo) |
| [E003](E003_passa-banda-1-20hz/README.md) | passa-banda-1-20hz | E002 | Filtro 1–20 Hz + normalização só com saudáveis | Sem a faixa de 64–128 Hz, o erro mede a banda modal | AUC sev. 0,5: 0,55 → 0,72; 1,0: 0,99; até 0,1 ainda ~0,5. Saudáveis acima do limiar: 16% → 5%, sem escolha | Adotada (fração não reconstruída, η² velocidade) | Concluído (retroativo) |
| [E004](E004_pooling-2x2x2/README.md) | pooling-2x2x2 | E003 | `pool_sizes` [4,4,2] → [2,2,2] (redução no tempo 32× → 8×), mantendo latente (912) e campo receptivo (≥ 128); 100 épocas | Menos redução no tempo reduz a energia criada fora da banda e o erro em 12–20 Hz | — | — | Proposto |
