# Experimentos: Autoencoder Conv1D Esparso

Fluxo e regras: `docs/experiments.md`. Modelo de registro: `docs/experiment_template.md`.

- **Base atual:** E005 (arquitetura do E004 com limiar no percentil 95; épocas da base, 100 ou 200, em aberto).
- **Variantes avaliadas nos folds de teste até aqui:** 5 (E001–E005).

| ID | Título | Pai | Mudança | Hipótese (resumo) | Resultado principal | Decisão | Status |
|---|---|---|---|---|---|---|---|
| [E001](E001_baseline-tempo/README.md) | baseline-tempo | — | Sinal no tempo, sem filtro, 100 épocas | AE treinado em saudáveis reconstrói pior passagens com dano | AUC ≈ 0,5 até sev. 0,5; 0,90 em 1,0; FPR 17% | Base inicial | Concluído (retroativo) |
| [E002](E002_200-epocas/README.md) | 200-epocas | E001 | `epochs` 100 → 200 | Mais épocas reduzem o erro dos saudáveis e o modelo converge | `val_loss` 0,330 → 0,312; AUC por severidade inalterada | Adotada (`val_loss`) | Concluído (retroativo) |
| [E003](E003_passa-banda-1-20hz/README.md) | passa-banda-1-20hz | E002 | Filtro 1–20 Hz + normalização só com saudáveis | Sem a faixa de 64–128 Hz, o erro mede a banda modal | AUC sev. 0,5: 0,55 → 0,72; 1,0: 0,99; até 0,1 ainda ~0,5. Saudáveis acima do limiar: 16% → 5%, sem escolha | Adotada (fração não reconstruída, η² velocidade) | Concluído (retroativo) |
| [E004](E004_pooling-2x2x2/README.md) | pooling-2x2x2 | E003 | `pool_sizes` [4,4,2] → [2,2,2] (redução no tempo 32× → 8×), mantendo latente (912) e campo receptivo (≥ 128); 100 épocas | Menos redução no tempo reduz a energia criada fora da banda e o erro em 12–20 Hz | `val_loss` 0,0907 → 0,0134 (contra E003 com 100 épocas); AUC sev. 0,2: 0,66; 0,5: 0,83; até 0,1 ainda ~0,5; FPR 1,6% | Adotada (resíduo fora da banda, `val_loss`) | Concluído |
| [E005](E005_limiar-percentil-95/README.md) | limiar-percentil-95 | E004 | Limiar μ + 1σ → percentil 95 dos saudáveis de treino (sem retreino) | O percentil fixa em 5% a fração de saudáveis de treino acima do limiar | 5% em todos os folds; FPR 8,6% (84 de 116 FP nos folds de peso 0,90); recall sev. 0,2: 18%, 0,5: 64%; AUC igual | Adotada (decisão de projeto: α = 5%) | Concluído |
