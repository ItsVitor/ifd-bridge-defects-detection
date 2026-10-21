# Experimentos: Autoencoder Conv1D Esparso

Fluxo e regras: `docs/experiments.md`. Modelo de registro: `docs/experiment_template.md`.

- **Base atual:** E001 (provisória até o registro de E002 e E003).
- **Variantes avaliadas nos folds de teste até aqui:** 3 (E001–E003).

| ID | Título | Pai | Mudança | Hipótese (resumo) | Resultado principal | Decisão | Status |
|---|---|---|---|---|---|---|---|
| [E001](E001_baseline-tempo/README.md) | baseline-tempo | — | Sinal no tempo, sem filtro, 100 épocas | AE treinado em saudáveis reconstrói pior passagens com dano | AUC ≈ 0,5 até sev. 0,5; 0,90 em 1,0; FPR 17% | Base inicial | Concluído (retroativo) |
| E002 | 200-epocas | E001 | `epochs` 100 → 200 | `val_loss` ainda caía na época 100 | Sem mudança relevante | — | Retroativo, a documentar |
| E003 | passa-banda-1-20hz | E002 | Filtro 1–20 Hz + correção `supervised` | Faixa de 64–128 Hz não reconstruída dominava o erro | AUC sev. 0,5: 0,55 → 0,73; FPR 18% → 11% | — | Retroativo, a documentar |
