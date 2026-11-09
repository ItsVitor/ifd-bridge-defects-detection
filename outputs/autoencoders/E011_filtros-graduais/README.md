# E011: Filtros graduais no encoder (32 → 16 → 4)

| Campo | Valor |
|---|---|
| Status | Proposto |
| Pai (base de comparação) | [E010](../E010_nyquist-latente-4-canais/README.md) |
| Proposto em | 2026-11-09 |
| Executado em | — |
| Código | — (commitar antes do run) |

## 1. Motivação

**A compressão do encoder está concentrada no último bloco.** Os filtros `[32, 32, 16]` vieram do E001 como pontos de partida arbitrários, e o último valor caiu para 4 no E004 só para manter o tamanho do latente. Valores que saem de cada bloco do encoder no E010 (base):

| Etapa | Formato | Valores | Redução |
|---|---|---|---|
| Entrada | 1824 × 16 | 29.184 | — |
| Bloco 1 (32 filtros, pool 2) | 912 × 32 | 29.184 | 1× |
| Bloco 2 (32 filtros, pool 2) | 456 × 32 | 14.592 | 2× |
| Bloco 3 (4 filtros, pool 1) | 456 × 4 | 1824 | **8×** |

A última convolução do encoder precisa passar de 32 para 4 canais de uma vez, e o decoder faz o mesmo salto ao contrário (4 → 32).

**Os canais mortos aparecem justamente nessa camada.** No E010, um dos 4 canais do latente ficou sempre em zero nos folds 1 e 4 (diagnóstico só com saudáveis de treino, `residual_spectrum.txt`). O mesmo aconteceu no E004, no E006 e no E009. Não há medição que ligue os canais mortos à queda abrupta; é só onde eles aparecem.

**Literatura (pesquisada a pedido do autor).**
- A distribuição mais comum em CNNs é a pirâmide (dobrar os filtros cada vez que o pooling divide a resolução por 2), mas ela é uma heurística: [Izquierdo-Cordova e Mayol-Cuevas (2021)](https://arxiv.org/html/2104.08446v1) mostraram que distribuições alternativas igualam ou superam a pirâmide com bem menos parâmetros.
- Em CNNs 1D para detecção de falhas, 16 ou 32 filtros por camada costumam ser a melhor escolha ([Hudson et al., 2024](https://arxiv.org/abs/2411.15191)).
- Nenhuma fonte encontrada justifica ou condena uma queda abrupta antes do gargalo.

## 2. Hipótese

> Se a compressão do encoder for dividida igualmente entre os blocos 2 e 3 (filtros `[32, 16, 4]`, 4× em cada bloco), mantendo o latente em 456 × 4 e o campo receptivo, então o erro dos saudáveis não aumenta e nenhum canal do latente morre, com 34% menos parâmetros, porque nenhuma camada precisa comprimir 8× de uma vez.

A hipótese é de parcimônia: o modelo gradual reconstrói tão bem quanto o abrupto com menos parâmetros. Ela é refutada se o MSE dos saudáveis subir ou se canais continuarem morrendo.

## 3. Mudança em relação ao pai

| Item | E010 | E011 |
|---|---|---|
| `encoder_filters` | `[32, 32, 4]` | `[32, 16, 4]` |

Sem mudanças de código. O decoder é o espelho do encoder, então ele também fica gradual (4 → 16 → 32, em vez de 4 → 32 → 32).

| Etapa | E010 | E011 |
|---|---|---|
| Bloco 1 | 912 × 32 = 29.184 (1×) | 912 × 32 = 29.184 (1×) |
| Bloco 2 | 456 × 32 = 14.592 (2×) | 456 × 16 = 7296 (4×) |
| Bloco 3 (latente) | 456 × 4 = 1824 (8×) | 456 × 4 = 1824 (4×) |

**Por que 16.** É o único valor que divide igualmente a compressão de 16× entre os blocos 2 e 3: de 29.184 para 7296 (4×) e de 7296 para 1824 (4×). A regra "dividir igualmente" é uma escolha de projeto, não uma medição. Ela também coincide com a faixa de 16–32 filtros que Hudson et al. encontraram em CNNs 1D de falhas.

**O que se mantém.** O latente continua com 456 × 4 = 1824 valores, e o campo receptivo do encoder continua em 130 amostras, porque ele só depende do kernel e dos pools.

**Efeito confundido: menos parâmetros.** O modelo cai de 63.688 para 41.768 parâmetros (−34%, contados com `build_model`). A diferença vem de duas camadas: a segunda convolução do encoder (19.488 → 9744) e a penúltima do decoder (19.488 → 9760); as camadas ligadas ao latente também diminuem um pouco (2436 → 1220 e 2464 → 1232). Se o erro subir, não dá para separar o efeito da distribuição dos filtros do efeito de ter menos capacidade. Manter os parâmetros iguais exigiria aumentar os filtros do primeiro bloco, o que mudaria mais uma coisa.

**Custo.** As duas camadas que encolhem rodam a 912 instantes, então o custo por época deve cair um pouco em relação ao E010.

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

Todas as medidas de espectro e latente vêm de `residual_spectrum.txt` (mediana e faixa dos 15 folds), comparáveis diretamente com o E010.

| Critério | E010 | Previsão para o E011 |
|---|---|---|
| Mediana, entre folds, do MSE médio dos saudáveis de treino | 0,0033 (0,0028–0,0047) | Igual ou até 10% maior (≤ 0,0036) |
| Canais ativos no latente | 4 de 4 em 13 folds; 3 de 4 nos folds 1 e 4 | 4 de 4 em todos os folds |
| Parte não reconstruída de 16–20 Hz | 3,1% (2,7–4,4%) | Sem previsão de direção |
| Energia criada fora da banda, relativa à entrada | 0,02% (0,01–0,03%) | Sem previsão de direção |
| Zeros no latente | 29,3% (26,3–32,1%) | Sem previsão |
| Penalidade L1 ÷ MSE | 0,38 (0,35–0,40) | Sem previsão |
| Erro do peso 0,90 ÷ outros pesos (mediana entre folds) | 1,27 (1,23–1,41) | ≈ 1,3 |
| Folds parados pelo early stopping | 15 de 15 | Maioria |

O `val_loss` usa o mesmo coeficiente L1 do E010 e pode ser comparado diretamente.

### 4.2 Avaliação (só reporte)

Sem previsão de direção para a AUC e os falsos negativos. Se a reconstrução dos saudáveis ficar igual, a detecção deve ficar perto da do E010 (FN 891, FPR 6,7%, AUC sev. 0,2 0,75). Severidades até 0,05 devem continuar com AUC ≈ 0,5.

### 4.3 Critério de adoção

Adotar `encoder_filters = [32, 16, 4]` na base se as duas condições valerem:
1. A mediana, entre folds, do MSE médio dos saudáveis de treino não for maior que a do E010 (0,0033, com 4 casas decimais).
2. A mediana, entre folds, de canais ativos no latente for 4 de 4.

A adoção se justifica por parcimônia: mesmo erro com 34% menos parâmetros. Se o MSE subir, o E010 continua a base, mesmo que os canais mortos desapareçam.

**Limite conhecido do critério.** O MSE dos saudáveis não antecipa a detecção (E002, E006, E007 e E008), e a comparação "não maior" depende de uma única semente, sem estimativa da variação entre treinos. A métrica de teste é só reportada e não decide.

## 5. Configuração e origem de cada escolha

Igual ao E010, exceto a linha marcada.

| Parâmetro | Valor | Origem | Justificativa / evidência |
|---|---|---|---|
| `nodes_to_use` | 8 nós (10654, 10659, 10664, 10669, 10658, 10663, 10668, 10673) | Herdado / Pedido | Mesmos nós dos pipelines OCSVM e Isolation Forest, para comparação direta |
| `axes_to_use` | X, Y | Herdado / Pedido | Idem |
| `filter_type`, `cutoff_low`, `cutoff_high` | Passa-banda 1–20 Hz (Butterworth ordem 4, `filtfilt`) | Medido / Físico | Ver E003 |
| Ordem filtro/recorte | Filtra o registro completo, depois recorta | Físico | Ver E003 |
| Comprimento do sinal | Recorte em 1824 amostras (7,12 s) | Medido / Físico | Ver E001 |
| Normalização | Padronização por canal, só com saudáveis do treino | Físico / premissa | Ver E003 |
| Seleção de features | Nenhuma | Físico | Colunas são instantes de tempo |
| `pool_sizes` | `[2, 2, 1]` | Físico / Medido | Adotado no E010 |
| ★ `encoder_filters` | `[32, 16, 4]` | Pedido / Arbitrário | Pedido do autor (filtros graduais). O 16 vem da regra de dividir a compressão igualmente entre os blocos 2 e 3 (seção 3); a regra é uma escolha de projeto. O 32 do primeiro bloco e o 4 do latente são herdados (E001 e E004) |
| `kernel_size` | 19 | Físico (heurística) | Adotado no E004; campo receptivo de 130 amostras |
| `l1_activity` | 1e-5 | Pedido / Arbitrário | Adotado no E008 |
| Ativações e camadas | ReLU; saída linear; `UpSampling1D` + Conv | Convenção | Ver E001. O autor preferiu não trocar a ativação do latente por enquanto |
| `learning_rate` | 1e-3 (Adam) | Convenção | Padrão do Adam |
| `epochs` | 200 | Medido / Pedido | Adotado no E006, com early stopping |
| `early_stopping_patience` | 10 (com `restore_best_weights`) | Convenção | Valor comum |
| `batch_size` | 32 | Convenção | Valor comum |
| `validation_split` | 0,1, após embaralhar os saudáveis | Convenção | Ver E001 |
| Score de anomalia | MSE médio sobre 1824 × 16 | Convenção | Erro de reconstrução usual; não inclui a penalidade |
| Regra do limiar | Percentil 95 do erro dos saudáveis de treino de cada fold | Pedido / Medido | Adotada no E005 (α = 5%) |
| `random_state` | 42 | Convenção | Semente fixa |

## 6. Verificação da premissa não supervisionada

- [x] Modelo, normalização, filtros e limiar ajustados só com saudáveis do treino.
- [x] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC. A mudança vem do pedido do autor, da distribuição da compressão entre as camadas e dos canais mortos medidos nos saudáveis.
- [x] Motivação e critério de adoção usam só saudáveis ou raciocínio de projeto.

## 7. Execução

Comando (depois de commitar esta pasta). O espectro do resíduo de todos os folds sai no mesmo run:

```powershell
.venv\Scripts\python.exe src\test_autoencoders.py outputs\autoencoders\E011_filtros-graduais *> outputs\autoencoders\E011_filtros-graduais\run.log
```

Antes de rodar em outra máquina: `uv sync`.

Duração estimada: o teto é de 3000 épocas (15 folds × 200). O E010 somou 2112 épocas (duração não registrada; o E009, com custo por época parecido, levou ~3h45 para 2398 épocas). O custo por época do E011 deve ser um pouco menor que o do E010 (seção 3), mas o ponto de parada pode mudar.

## 8. Resultados

## 9. Análise

## 10. Decisão e próximos passos

## Notas posteriores
