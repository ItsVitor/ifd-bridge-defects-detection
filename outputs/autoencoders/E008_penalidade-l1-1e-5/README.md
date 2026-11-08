# E008: Penalidade L1 de 1e-5 no latente

| Campo | Valor |
|---|---|
| Status | Proposto |
| Pai (base de comparação) | [E006](../E006_200-epocas-pooling-2x2x2/README.md) |
| Proposto em | 2026-11-08 |
| Executado em | — |
| Código | — (commitar antes do run) |

## 1. Motivação

O E007 mostrou, só com saudáveis, que a penalidade `l1_activity = 1e-4` do E006 era forte demais para esta arquitetura:

| Medição (saudáveis) | E006 (L1 = 1e-4) | E007 (L1 = 0) |
|---|---|---|
| Mediana, entre folds, do MSE médio dos saudáveis de treino | 0,0077 | 0,0050 |
| Canais ativos no latente (fold 1) | 2 de 4 | 4 de 4 |
| Zeros no latente (fold 1) | 53,3% | 22,7% |
| Penalidade L1 ÷ MSE (fold 1) | 0,29 | 0 |
| Parte não reconstruída de 16–20 Hz (fold 1) | 7,1% | 5,2% |

- Era a penalidade que matava canais: sem ela, os 4 canais ficaram ativos.
- A ReLU sozinha produz ~23% de zeros; os outros ~30 pontos percentuais de zeros do E006 vinham da L1.

**Por que não adotar o E007.** O projeto pede um autoencoder Conv1D esparso, e o autor decidiu manter alguma penalidade L1, para que a esparsidade continue sendo buscada pelo treino. O E007 fica como diagnóstico e não é adotado; a base continua sendo o E006.

**Por que 1e-5.** É um décimo do coeficiente do E006. Se as ativações do latente tivessem o mesmo tamanho do E006, a penalidade cairia de 29% para ~3% do MSE, abaixo dos 10% do E003 (o único ponto em que a penalidade era pequena perto do MSE). O valor é um passo de uma ordem de grandeza, escolhido pelo autor; não há medição que aponte 1e-5 em vez de, por exemplo, 3e-5.

## 2. Hipótese

> Se a penalidade L1 cair de 1e-4 para 1e-5, então nenhum canal do latente morre e o erro de reconstrução dos saudáveis fica abaixo do E006, enquanto o latente fica mais esparso que no E007, porque a penalidade passa a pesar poucos por cento do MSE: o suficiente para empurrar ativações pequenas a zero, mas não para desligar canais inteiros.

## 3. Mudança em relação ao pai

| Item | E006 | E008 |
|---|---|---|
| `l1_activity` | 1e-4 | 1e-5 |

Sem mudanças de código. Arquitetura, épocas, early stopping e limiar (percentil 95) iguais aos do E006.

**Comparação de perdas.** Como o coeficiente muda, o `val_loss` do Keras (que inclui a penalidade) não é comparável com o do E006 nem com o do E007. A comparação de reconstrução usa o MSE puro de `train_errors.parquet`, como no E007.

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

| Critério | E006 | E007 | Previsão para o E008 |
|---|---|---|---|
| Mediana, entre folds, do MSE médio dos saudáveis de treino | 0,0077 | 0,0050 | Entre os dois, mais perto do E007 |
| Canais ativos no latente (fold 1) | 2 de 4 | 4 de 4 | 4 de 4 |
| Zeros no latente (fold 1) | 53,3% | 22,7% | Entre os dois |
| Penalidade L1 ÷ MSE (fold 1) | 0,29 | 0 | Entre 0,03 e 0,10 (coeficiente 10× menor, compensado em parte por ativações maiores) |
| Parte não reconstruída de 16–20 Hz, fold 1 | 7,1% | 5,2% | Entre os dois; ainda a pior faixa |
| Erro do peso 0,90 ÷ outros pesos | 1,31 | 1,35 | Sem mudança (≈ 1,3) |
| Folds parados pelo early stopping | 15 de 15 | 15 de 15 | 15 de 15 |

### 4.2 Avaliação (só reporte)

O E007 mudou pouco a detecção (FN 886 → 886; AUC sev. 0,2: 0,68 → 0,70; sev. 0,5: 0,835 → 0,855). Como o E008 fica entre E006 e E007, a previsão é que a detecção também fique entre os dois: FN perto de 886 e AUC por severidade dentro do desvio entre folds. Severidades até 0,1 continuam com AUC ≈ 0,5. O limiar continua decidido pelo peso 0,90, então não se espera queda de falsos negativos.

### 4.3 Critério de adoção

Adotar `l1_activity = 1e-5` na base se as duas condições valerem:
1. A mediana, entre folds, do MSE médio dos saudáveis de treino for menor que a do E006 (0,0077).
2. O fold 1 tiver os 4 canais do latente ativos.

Se a condição 2 falhar (algum canal morto), a penalidade ainda é forte demais para esta arquitetura e o próximo passo é um coeficiente menor. Se só a 1 falhar, a L1 de 1e-4 permanece.

**Limite conhecido do critério.** Menor MSE dos saudáveis não garante melhor detecção (E002, E006 e E007 mostraram isso). A métrica de teste é só reportada e não decide.

## 5. Configuração e origem de cada escolha

Igual ao E006, exceto a linha marcada.

| Parâmetro | Valor | Origem | Justificativa / evidência |
|---|---|---|---|
| `nodes_to_use` | 8 nós (10654, 10659, 10664, 10669, 10658, 10663, 10668, 10673) | Herdado / Pedido | Mesmos nós dos pipelines OCSVM e Isolation Forest, para comparação direta |
| `axes_to_use` | X, Y | Herdado / Pedido | Idem |
| `filter_type`, `cutoff_low`, `cutoff_high` | Passa-banda 1–20 Hz (Butterworth ordem 4, `filtfilt`) | Medido / Físico | Ver E003 |
| Ordem filtro/recorte | Filtra o registro completo, depois recorta | Físico | Ver E003 |
| Comprimento do sinal | Recorte em 1824 amostras (7,12 s) | Medido / Físico | Ver E001 |
| Normalização | Padronização por canal, só com saudáveis do treino | Físico / premissa | Ver E003 |
| Seleção de features | Nenhuma | Físico | Colunas são instantes de tempo |
| `pool_sizes` | `[2, 2, 2]` | Medido / Físico | Adotado no E004 |
| `encoder_filters` | `[32, 32, 4]` | Arbitrário / controle | Adotado no E004 |
| `kernel_size` | 19 | Físico (heurística) | Adotado no E004 |
| ★ `l1_activity` | 1e-5 | Pedido / Arbitrário | O autor quer manter a penalidade (seção 1). A ordem de grandeza menor vem do E007 (canais mortos e penalidade em 29% do MSE com 1e-4); o valor exato 1e-5 é arbitrário |
| Ativações e camadas | ReLU; saída linear; `UpSampling1D` + Conv | Convenção | Ver E001 |
| `learning_rate` | 1e-3 (Adam) | Convenção | Padrão do Adam |
| `epochs` | 200 | Medido / Pedido | Adotado no E006, com early stopping |
| `early_stopping_patience` | 10 (com `restore_best_weights`) | Convenção | Monitora o `val_loss`, que inclui a penalidade (pequena com 1e-5) |
| `batch_size` | 32 | Convenção | Valor comum |
| `validation_split` | 0,1, após embaralhar os saudáveis | Convenção | Ver E001 |
| Score de anomalia | MSE médio sobre 1824 × 16 | Convenção | Erro de reconstrução usual; não inclui a penalidade |
| Regra do limiar | Percentil 95 do erro dos saudáveis de treino de cada fold | Pedido / Medido | Adotada no E005 (α = 5%) |
| `random_state` | 42 | Convenção | Semente fixa |

## 6. Verificação da premissa não supervisionada

- [x] Modelo, normalização, filtros e limiar ajustados só com saudáveis do treino.
- [x] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC. O coeficiente vem da decisão do autor e das medições do latente nos saudáveis (E006, E007).
- [x] Motivação e critério de adoção usam só saudáveis ou raciocínio físico.

## 7. Execução

Comandos (depois de commitar esta pasta):

```powershell
.venv\Scripts\python.exe src\test_autoencoders.py outputs\autoencoders\E008_penalidade-l1-1e-5 *> outputs\autoencoders\E008_penalidade-l1-1e-5\run.log
.venv\Scripts\python.exe src\residual_spectrum.py outputs\autoencoders\E008_penalidade-l1-1e-5 *> outputs\autoencoders\E008_penalidade-l1-1e-5\residual_spectrum.log
```

Antes de rodar em outra máquina: `uv sync`.

Duração estimada: o teto é de 3000 épocas (15 folds × 200), mas o early stopping parou antes nos pais: 2072 épocas no E006 (~6h20 na outra máquina) e 1828 no E007. O diagnóstico do resíduo leva ~20–30 min.

**Limitação:** canais ativos, zeros e o espectro do resíduo são medidos só no fold 1. O critério 4.3 (2) depende dessa medição única.

## 8. Resultados

## 9. Análise

## 10. Decisão e próximos passos

## Notas posteriores
