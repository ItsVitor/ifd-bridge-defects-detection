# E007: Sem penalidade L1 no latente

| Campo | Valor |
|---|---|
| Status | Proposto |
| Pai (base de comparação) | [E006](../E006_200-epocas-pooling-2x2x2/README.md) |
| Proposto em | 2026-11-08 |
| Executado em | — |
| Código | — (commitar antes do run) |

## 1. Motivação

O coeficiente `l1_activity = 1e-4` é um ponto de partida arbitrário do E001 e nunca foi testado. Três medições, todas só com saudáveis, indicam que ele pesa mais do que se pretendia na arquitetura atual:

| Medição | E003 | E004 | E006 |
|---|---|---|---|
| Penalidade L1 ÷ MSE (fold 1, `residual_spectrum.txt`) | 0,10 | 0,29 | 0,29 |
| Canais ativos no latente (fold 1) | 14 de 16 | 3 de 4 | 2 de 4 |
| Zeros no latente (fold 1) | 88,6% | 54,4% | 53,3% |

- **A penalidade triplicou em relação ao MSE.** Da E003 para a E004, o MSE caiu ~8× e a penalidade não acompanhou. A intenção do E001 era uma penalidade pequena perto do MSE.
- **Canais estão morrendo.** No E006, metade do latente (2 de 4 canais) está sempre em zero no fold 1. O latente útil cai de 912 para ~456 valores. Mais épocas (E004 → E006) mataram mais um canal.
- **Não se sabe quanto da esparsidade vem da L1.** A ReLU, sozinha, já produz zeros exatos. Os 53% de zeros podem ser, em parte ou quase todos, efeito da ReLU.

**A perda reportada pelo Keras inclui a penalidade.** Medido no `run.log` e nos parquets do E006: a perda de treino na melhor época é 1,22–1,38× o MSE puro dos saudáveis de treino (mediana do MSE 0,0077; da perda 0,0102). Por isso, o `val_loss` não serve para comparar o E006 com o E007: sem a L1, ele cairia ~25% mesmo com a reconstrução igual. As comparações entre E002 e E006 foram justas, porque todos usavam o mesmo coeficiente.

**Por que 0, e não um valor menor como 1e-5.** O objetivo principal é medir quanto da esparsidade e dos canais mortos vem da penalidade. Com 0, a resposta é direta: o que sobrar de zeros é da ReLU. Um valor intermediário responderia "um pouco menos de penalidade" e não separaria os dois efeitos.

## 2. Hipótese

> Se a penalidade L1 for removida (`l1_activity = 0`), então os canais mortos do latente voltam a ser usados e o erro de reconstrução dos saudáveis cai, porque hoje a penalidade, equivalente a ~29% do MSE, empurra ativações úteis para zero e mata canais inteiros.

## 3. Mudança em relação ao pai

| Item | E006 | E007 |
|---|---|---|
| `l1_activity` | 1e-4 | 0 |

Sem mudanças de código. A camada `ActivityRegularization` continua no modelo, mas com coeficiente 0 ela não acrescenta nada à perda; a arquitetura, os parâmetros e o limiar (percentil 95) são os mesmos do E006.

**O modelo deixa de ser "esparso" no sentido do pedido inicial.** O projeto pediu um autoencoder Conv1D esparso, e a penalidade L1 é o que torna o latente esparso por objetivo. Sem ela, o latente ainda tem zeros (pela ReLU), mas a esparsidade deixa de ser buscada pelo treino. Se o resultado indicar adoção, isso precisa ser uma decisão explícita do autor (seção 4.3).

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

O critério de reconstrução é o **MSE puro**, calculado a partir de `train_errors.parquet` (média por fold dos saudáveis de treino), e não o `val_loss`.

| Critério | E006 | Previsão para o E007 |
|---|---|---|
| Mediana, entre folds, do MSE médio dos saudáveis de treino | 0,0077 | Menor (mais capacidade útil no latente) |
| Canais ativos no latente (fold 1) | 2 de 4 | 4 de 4 |
| Zeros no latente (fold 1) | 53,3% | Menor, mas acima de 0% (a ReLU continua) |
| Penalidade L1 ÷ MSE (fold 1) | 0,29 | 0 (por construção) |
| Erro relativo total, fold 1 | 0,64% | Menor |
| Parte não reconstruída de 16–20 Hz, fold 1 | 7,1% | Menor; ainda a pior faixa |
| Erro do peso 0,90 ÷ outros pesos | 1,31 | Sem previsão de direção |
| Folds parados pelo early stopping | 15 de 15 | Maioria |

### 4.2 Avaliação (só reporte)

Sem previsão de direção para a AUC e os falsos negativos. Há dois cenários plausíveis:
- **Mais capacidade ajuda a separar:** como no E004, o modelo passa a reproduzir com mais detalhe a estrutura saudável, e o dano fica relativamente pior reconstruído. AUC das severidades 0,2 e 0,5 sobe.
- **Mais capacidade reconstrói também o dano:** o modelo fica mais perto de copiar a entrada, e a diferença entre saudáveis e danificados diminui. AUC cai.

Severidades até 0,1 devem continuar com AUC ≈ 0,5.

### 4.3 Critério de adoção

Adotar `l1_activity = 0` na base se as duas condições valerem:
1. A mediana, entre folds, do MSE médio dos saudáveis de treino for menor que a do E006 (0,0077).
2. O fold 1 tiver mais canais ativos no latente que o E006 (mais que 2 de 4).

Como a adoção remove a esparsidade buscada pelo treino, que fazia parte do pedido inicial, ela também depende de confirmação do autor. Se o critério não for atendido, a L1 permanece e o resultado é registrado como diagnóstico (quanto da esparsidade vem da ReLU).

**Limite conhecido do critério.** Menor MSE dos saudáveis não garante melhor detecção (E002 e E006 mostraram isso). Não existe critério só com saudáveis que antecipe se mais capacidade vai separar mais ou reconstruir também o dano. A métrica de teste é só reportada e não decide.

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
| ★ `l1_activity` | 0 | Medido | Penalidade em 29% do MSE e 2 de 4 canais mortos no E006 (seção 1). Valor 0 para isolar o efeito da penalidade |
| Ativações e camadas | ReLU; saída linear; `UpSampling1D` + Conv | Convenção | Ver E001 |
| `learning_rate` | 1e-3 (Adam) | Convenção | Padrão do Adam |
| `epochs` | 200 | Medido / Pedido | Adotado no E006, com early stopping |
| `early_stopping_patience` | 10 (com `restore_best_weights`) | Convenção | Valor comum. Monitora o `val_loss`, que, com L1 = 0, passa a ser o MSE puro de validação |
| `batch_size` | 32 | Convenção | Valor comum |
| `validation_split` | 0,1, após embaralhar os saudáveis | Convenção | Ver E001 |
| Score de anomalia | MSE médio sobre 1824 × 16 | Convenção | Erro de reconstrução usual; nunca incluiu a penalidade |
| Regra do limiar | Percentil 95 do erro dos saudáveis de treino de cada fold | Pedido / Medido | Adotada no E005 (α = 5%) |
| `random_state` | 42 | Convenção | Semente fixa |

## 6. Verificação da premissa não supervisionada

- [x] Modelo, normalização, filtros e limiar ajustados só com saudáveis do treino.
- [x] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC. O coeficiente 0 vem das medições do latente e da penalidade nos saudáveis.
- [x] Motivação e critério de adoção usam só saudáveis ou raciocínio físico.

## 7. Execução

Comandos (depois de commitar esta pasta):

```powershell
.venv\Scripts\python.exe src\test_autoencoders.py outputs\autoencoders\E007_sem-penalidade-l1 *> outputs\autoencoders\E007_sem-penalidade-l1\run.log
.venv\Scripts\python.exe src\residual_spectrum.py outputs\autoencoders\E007_sem-penalidade-l1 *> outputs\autoencoders\E007_sem-penalidade-l1\residual_spectrum.log
```

Antes de rodar em outra máquina: `uv sync`, para instalar o `matplotlib` declarado no `pyproject.toml`.

Duração estimada: parecida com a do E006, que levou ~6h20 na outra máquina. O teto é de 3000 épocas no total (15 folds × 200 épocas), mas o early stopping costuma parar antes: no E006, os 15 folds somaram 2072 épocas (paradas entre as épocas 85 e 189). Sem a L1, o ponto de parada pode mudar, então a duração real pode ficar acima ou abaixo da do E006. O diagnóstico do resíduo leva ~20–30 min.

**Limitação:** canais ativos, zeros e o espectro do resíduo são medidos só no fold 1 (ver discussão no E006). O critério 4.3 (2) depende dessa medição única.

## 8. Resultados

## 9. Análise

## 10. Decisão e próximos passos

## Notas posteriores
