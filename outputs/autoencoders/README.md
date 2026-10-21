# Experimentos: Autoencoder Conv1D Esparso

Fluxo e regras: `docs/experiments.md`. Modelo de registro: `docs/experiment_template.md`.

- **Base atual:** E002 (provisória até o registro do E003).
- **Variantes avaliadas nos folds de teste até aqui:** 3 (E001–E003).

| ID | Título | Pai | Mudança | Hipótese (resumo) | Resultado principal | Decisão | Status |
|---|---|---|---|---|---|---|---|
| [E001](E001_baseline-tempo/README.md) | baseline-tempo | — | Sinal no tempo, sem filtro, 100 épocas | AE treinado em saudáveis reconstrói pior passagens com dano | AUC ≈ 0,5 até sev. 0,5; 0,90 em 1,0; FPR 17% | Base inicial | Concluído (retroativo) |
| [E002](E002_200-epocas/README.md) | 200-epocas | E001 | `epochs` 100 → 200 | Mais épocas reduzem o erro dos saudáveis e o modelo converge | `val_loss` 0,330 → 0,312; AUC por severidade inalterada | Adotada (`val_loss`) | Concluído (retroativo) |
| E003 | passa-banda-1-20hz | E002 | Filtro 1–20 Hz + correção `supervised` | Faixa de 64–128 Hz não reconstruída dominava o erro | AUC sev. 0,5: 0,55 → 0,73; FPR 18% → 11% | — | Retroativo, a documentar |
