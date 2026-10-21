# E005: Limiar no percentil 95 dos saudáveis de treino

| Campo | Valor |
|---|---|
| Status | Proposto |
| Pai (base de comparação) | [E004](../E004_pooling-2x2x2/README.md) |
| Proposto em | 2026-10-21 |
| Executado em | — |
| Código | — (commitar antes do run) |

## 1. Motivação

A regra do limiar μ + 1σ não define a fração de alarmes falsos: essa fração depende do formato da distribuição do erro dos saudáveis. Com a mesma regra, a fração de saudáveis de treino acima do limiar foi de 15,9% (E001), 16,3% (E002), 5,2% (E003) e 1,4% (E004), sem que ninguém escolhesse esses valores.

No E004, a causa é um grupo de 15 passagens saudáveis (irregularidade 2 × módulo de elasticidade 0,975) com erro 7–11× maior que o resto (ver E004, Notas posteriores). Esse grupo infla o σ, e a assimetria do erro chega a 9,6. Medido só nos saudáveis de treino do E004:

| Regra | Limiar (faixa entre folds) | Saudáveis de treino acima |
|---|---|---|
| μ + 1σ (E004) | 0,0166–0,0215 | 1,4% (irregularidade 1: 0,3%) |
| Percentil 99 | ~0,021 em 5 folds; 0,056–0,073 nos outros 10 | 1% |
| Percentil 95 | 0,0121–0,0149 | 5% |

O percentil 99 foi descartado antes de rodar, porque o 1% superior (~13 passagens) é praticamente o grupo das 15. O valor dele salta de patamar conforme 12 ou 15 dessas passagens caem no treino do fold, e o intervalo de 95% por bootstrap no fold 1 vai de 0,016 a 0,079. O percentil 95 fica abaixo do grupo e é estável entre folds.

## 2. Hipótese

> Se o limiar de cada fold for o percentil 95 do erro dos saudáveis de treino, então a fração de saudáveis de treino acima do limiar passa a ser 5% em todos os folds, independentemente do formato da distribuição, porque o percentil é definido diretamente sobre essa fração.

A hipótese vale por construção. O objetivo do experimento é medir e reportar o efeito de fixar o α em 5% no detector do E004, cujo score separa as severidades 0,2 e 0,5 melhor do que o limiar μ + 1σ aproveitava.

## 3. Mudança em relação ao pai

| Item | E004 | E005 |
|---|---|---|
| Regra do limiar | μ + 1σ dos saudáveis de treino | Percentil 95 dos saudáveis de treino (`threshold_rule: "percentile"`, `threshold_percentile: 95`) |

Nada mais muda: o modelo, o pré-processamento e os erros de reconstrução são os do E004.

Mudanças de código:
- `src/autoencoders_pipeline.py`: função `compute_threshold` com as regras `"mean_std"` e `"percentile"`, lidas do config. Configs anteriores, sem `threshold_rule`, continuam usando μ + kσ.
- `src/rethreshold.py`: reavalia um experimento com outra regra de limiar a partir dos erros salvos do pai, sem retreinar. Ele recusa configs que difiram do pai em algo além das chaves do limiar.
- `src/test_autoencoders.py`: o relatório por severidade e o diagnóstico viraram a função `print_error_report`, usada pelos dois scripts. A saída não muda.

**Por que não retreinar.** O erro de reconstrução de cada amostra não depende do limiar, e o treino é determinístico. Um run completo reproduziria os mesmos erros do E004 e só mudaria o limiar. Verificado antes do run: reavaliar o E004 com a própria regra (μ + 1σ) pelo `rethreshold.py` reproduz exatamente os limiares, as predições e as métricas do E004.

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

| Critério | E004 | Previsão para o E005 |
|---|---|---|
| Saudáveis de treino acima do limiar | 1,4% (1,0%–1,7% por fold) | 5,0% em todos os folds (63 de 1260) |
| Limiar por fold | 0,0166–0,0215 | 0,0121–0,0149 |
| Passagens do grupo irregularidade 2 × módulo 0,975 acima do limiar, no treino | todas | todas (12 ou 15 por fold, das 63 acima) |

### 4.2 Avaliação (só reporte)

- **AUC: idêntica à do E004, por construção.** Os scores são os mesmos; só o limiar muda.
- **FPR de teste: entre 4% e 8%.** Deve ficar perto de 5%, mas o erro dos saudáveis de teste não tem a mesma distribuição do treino em todos os folds:
  - nos folds de peso 0,90 (10–12), o erro mediano dos saudáveis de teste é ~30% maior que o dos outros pesos, então o FPR deve ser maior ali;
  - nos folds com módulo 0,975 no teste (1, 4, 7, 10 e 13), as 3 passagens do grupo irregularidade 2 × módulo 0,975 de cada um devem virar falsos positivos (+3,3 pontos de FPR nesses folds).
- **Recall:** maior que no E004 nas severidades 0,2 e 0,5. Para a 0,5, acima dos 53% do E004.
- **Severidades até 0,1:** recall perto do FPR, porque a AUC delas é ~0,5.
- **Severidade 1,0:** recall continua 100%.

### 4.3 Critério de adoção

A escolha do α = 5% é do autor do projeto (Pedido), baseada na estabilidade da estimativa nos saudáveis de treino (seção 1). O critério só com saudáveis é a regra entregar a fração escolhida: 5% dos saudáveis de treino acima do limiar em todos os folds. Ele vale por construção, então a adoção é, na prática, uma decisão de projeto. As métricas de teste são reportadas e não entram nessa decisão.

## 5. Configuração e origem de cada escolha

Igual ao E004, exceto a linha marcada.

| Parâmetro | Valor | Origem | Justificativa / evidência |
|---|---|---|---|
| `nodes_to_use` | 8 nós (10654, 10659, 10664, 10669, 10658, 10663, 10668, 10673) | Herdado | Mesmos nós dos configs de OCSVM e Isolation Forest |
| `axes_to_use` | X, Y | Herdado | Mesmos eixos dos outros pipelines |
| `filter_type`, `cutoff_low`, `cutoff_high` | Passa-banda 1–20 Hz (Butterworth ordem 4, `filtfilt`) | Medido / Físico | Ver E003 |
| Ordem filtro/recorte | Filtra o registro completo, depois recorta | Físico | Ver E003 |
| Comprimento do sinal | Recorte em 1824 amostras (7,12 s) | Medido / Físico | Ver E001 |
| Normalização | Padronização por canal, só com saudáveis do treino | Físico / premissa | Ver E003 |
| Seleção de features | Nenhuma | Físico | Colunas são instantes de tempo |
| `pool_sizes` | `[2, 2, 2]` | Medido / Físico | Adotado no E004 |
| `encoder_filters` | `[32, 32, 4]` | Arbitrário / controle | Adotado no E004 |
| `kernel_size` | 19 | Físico (heurística) | Adotado no E004 |
| `l1_activity` | 1e-4 | Arbitrário | Ponto de partida do E001. Medido no E004: 29% do MSE |
| Ativações e camadas | ReLU; saída linear; `UpSampling1D` + Conv | Convenção | Ver E001 |
| `learning_rate` | 1e-3 (Adam) | Convenção | Padrão do Adam |
| `epochs` | 100 | Pedido | Mantido do E004, porque os erros são reaproveitados |
| `early_stopping_patience` | 10 (com `restore_best_weights`) | Convenção | Valor comum |
| `batch_size` | 32 | Convenção | Valor comum |
| `validation_split` | 0,1, após embaralhar os saudáveis | Convenção | Ver E001 |
| Score de anomalia | MSE médio sobre 1824 × 16 | Convenção | Erro de reconstrução usual |
| ★ Regra do limiar | Percentil 95 do erro dos saudáveis de treino de cada fold | Pedido / Medido | α = 5% escolhido pelo autor. Percentil 99 descartado por instabilidade medida nos saudáveis (seção 1). Calibrado nos saudáveis de treino, e não na validação: com 126 amostras de validação, o percentil seria muito instável; e no E004 o modelo quase não sobreajusta (fold 1: perda de treino 0,0120, `val_loss` 0,0121) |
| `random_state` | 42 | Convenção | Semente fixa |

## 6. Verificação da premissa não supervisionada

- [x] Modelo, normalização, filtros e limiar ajustados só com saudáveis do treino. O limiar é um percentil dos erros dos saudáveis de treino de cada fold.
- [x] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC. O α foi escolhido pelo autor; o descarte do percentil 99 usou só a estabilidade nos saudáveis de treino.
- [x] Motivação e critério de adoção usam só saudáveis ou raciocínio físico.

## 7. Execução

Comando (depois de commitar o código e esta pasta), em menos de um minuto:

```powershell
.venv\Scripts\python.exe src\rethreshold.py outputs\autoencoders\E004_pooling-2x2x2 outputs\autoencoders\E005_limiar-percentil-95 *> outputs\autoencoders\E005_limiar-percentil-95\run.log
```

## 8. Resultados

## 9. Análise

## 10. Decisão e próximos passos

## Notas posteriores
