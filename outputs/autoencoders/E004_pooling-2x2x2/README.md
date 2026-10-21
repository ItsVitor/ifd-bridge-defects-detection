# E004: Menos redução no tempo (pooling 2 × 2 × 2)

| Campo | Valor |
|---|---|
| Status | Proposto |
| Pai (base de comparação) | [E003](../E003_passa-banda-1-20hz/README.md) |
| Proposto em | 2026-10-21 |
| Executado em | — |
| Código | — (commitar antes do run) |

## 1. Motivação

No E003, os valores de `pool_sizes`, `kernel_size` e `encoder_filters` continuam sendo os pontos de partida arbitrários do E001. Com a entrada agora limitada a 1–20 Hz, eles podem ser confrontados com a banda do sinal. Duas evidências, ambas só com saudáveis:

**1. O espaço latente tem resolução temporal muito abaixo da banda.** A redução no tempo é de 32× (`[4, 4, 2]`): o latente tem 57 instantes em 7,12 s, ou 8 amostras por segundo. A banda de entrada vai até 20 Hz. O decoder precisa recriar oscilações de 8–20 Hz a partir de um sinal amostrado a 8 Hz, ampliando ×4, ×4 e ×2 com `UpSampling1D`, que repete valores e gera "degraus".

**2. O decoder cria energia fora da banda.** O espectro do resíduo de reconstrução do E003 (fold 1, só saudáveis de treino, medido com `src/residual_spectrum.py`; ver `../E003_passa-banda-1-20hz/residual_spectrum.txt`):

| Faixa (Hz) | Fração da entrada | Fração do resíduo | Parte da faixa não reconstruída |
|---|---|---|---|
| 1–16 | 96,7% | 83,4% | 3–6% |
| 16–20 | 0,7% | 5,7% | 37% |
| 20–128 | ~0% | 8,0% | (sem entrada: energia criada pelo modelo) |

- O erro relativo total é 4,4%.
- Dentro de 1–16 Hz, o modelo já reconstrói 94–97% da energia.
- **8% do resíduo está acima de 20 Hz, onde a entrada não tem energia nenhuma.** Essa parte do score é criada pelo próprio decoder. Os "degraus" do `UpSampling1D` são a explicação mais provável: um degrau tem energia em alta frequência, e degraus maiores (×4) espalham mais energia. Essa explicação não foi verificada.
- A faixa de 16–20 Hz, perto da resolução que o latente consegue sustentar, é a mais mal reconstruída da banda (37%).

Outras medições do mesmo diagnóstico: 88,6% das ativações do latente são zero (14 dos 16 canais ativos), e a penalidade L1 equivale a 10% do MSE.

## 2. Hipótese

> Se a redução no tempo cair de 32× para 8× (`[2, 2, 2]`), mantendo a compressão total em 32× (latente de 912 valores) e o campo receptivo do encoder, então a energia criada fora da banda e o erro em 12–20 Hz diminuem, porque o decoder passa a ampliar o sinal em passos de ×2 a partir de um latente a 32 amostras/s, e não em passos de ×4 a partir de 8 amostras/s.

## 3. Mudança em relação ao pai

| Item | E003 | E004 |
|---|---|---|
| `pool_sizes` | `[4, 4, 2]` (redução de 32× no tempo; latente a 8 amostras/s) | `[2, 2, 2]` (redução de 8× no tempo; latente a 32 amostras/s) |
| `encoder_filters` | `[32, 32, 16]` | `[32, 32, 4]` |
| `kernel_size` | 9 | 19 |
| `epochs` | 200 | 100 |

Sem mudanças de código no modelo.

### Por que a arquitetura muda em três parâmetros

A mudança testada é uma só: onde a compressão acontece, no tempo ou nos canais. O tamanho do latente é o número de instantes vezes o número de canais do latente, que é o número de filtros da última camada do encoder (o último valor de `encoder_filters`):

| | Instantes no latente | Canais do latente | Total | Compressão total |
|---|---|---|---|---|
| E003 | 1824 ÷ 32 = 57 | 16 | 912 | 29.184 ÷ 912 = 32× |
| Só trocando os pools | 1824 ÷ 8 = 228 | 16 | 3648 | 8× |
| E004 | 1824 ÷ 8 = 228 | 4 | 912 | 32× |

Os outros dois parâmetros mudam para manter constantes duas grandezas que, de outro modo, mudariam junto e confundiriam o resultado:

| Grandeza mantida | E003 | E004 | Como |
|---|---|---|---|
| Tamanho do latente (compressão total) | 912 valores (32×) | 912 valores (32×) | Canais do latente: 16 → 4 |
| Campo receptivo do encoder ≥ 2 períodos do modo mais baixo (~4 Hz, 128 amostras) | 200 amostras (0,78 s) | 134 amostras (0,52 s) | `kernel_size`: 9 → 19 |

- Sem reduzir os canais do latente, o latente ficaria 4× maior. Isso mudaria também quanto o modelo comprime, e menos compressão pode fazer o modelo reconstruir também o dano.
- Sem aumentar o kernel, o campo receptivo cairia para 64 amostras (0,25 s), só um período do modo de ~4 Hz.
- O campo receptivo não fica idêntico (200 → 134), só acima do mesmo critério. Não dá para igualar os dois com pools de 2 e um kernel único sem uma camada a mais.

**Efeito confundido: menos canais no latente.** Cada instante do latente passa a ter 4 valores, em vez de 16, para descrever o estado dos 16 canais de entrada. É uma aposta de que as formas modais (como os nós vibram juntos) cabem em poucos canais. Se o resultado piorar, isso pode vir da falta de canais, e não da resolução no tempo; os dois efeitos não podem ser separados neste experimento.

### Por que 100 épocas

Pedido do autor: o custo por época fica ~2,8× maior, e 100 épocas permitem avaliar o resultado mais rápido. É uma segunda mudança, e ela interfere na comparação: no E003, ir de 100 para 200 épocas reduziu a mediana do melhor `val_loss` de 0,0907 para 0,0726 (−20%).

Para comparar só a arquitetura, as referências do E003 passam a ser as de 100 épocas:
- **`val_loss`:** o melhor valor de cada fold do E003 até a época 100, lido do `run.log` do E003. É uma referência exata: o treino é determinístico, e nenhum fold do E003 parou antes da época 100 (a primeira parada foi na 121). Mediana: 0,0907.
- **Espectro do resíduo:** o fold 1 do E003 foi retreinado com 100 épocas (`residual_spectrum_ref_E003_100ep.txt`).
- **AUC e demais métricas de teste:** não há run do E003 com 100 épocas. A comparação de AUC entre E003 e E004 fica confundida com as épocas, mas isso não afeta o critério de adoção, que usa só as duas referências acima.

**Diferença em relação ao "experimento B" discutido antes do E003.** O B previa kernel ~13 e redução de 4–8×, com latente de ~1824 valores. Refazendo as contas para este registro:
- Com pools `[2, 2, 2]` e kernel 13, o campo receptivo do encoder seria 92 amostras (0,36 s), abaixo de 2 períodos do modo de ~4 Hz.
- Um latente de 1824 valores dobraria o tamanho do latente, que é uma segunda mudança, com efeito próprio.

Por isso o kernel foi para 19 (que também atende a regra de cobrir ≥ 1 período da maior frequência da banda, 13 amostras) e o latente ficou em 912.

**Outros efeitos colaterais conhecidos:**
- Parâmetros: 39.344 → 63.688. Custo de cálculo por época: ~2,8× (26 → 73 milhões de multiplicações por amostra).
- O peso relativo da penalidade L1 pode mudar, porque a magnitude das ativações do latente muda. Será medido.

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

Referência: E003 com 100 épocas (ver seção 3). Medidos no fold 1 com `src/residual_spectrum.py`, salvo onde indicado:

| Critério | E003, 100 épocas | E003, 200 épocas | Previsão para o E004 (100 épocas) |
|---|---|---|---|
| Fração do resíduo acima de 20 Hz | 6,7% | 8,0% | Menor que 3,5% (cerca de metade) |
| Parte não reconstruída de 12–16 Hz | 9,0% | 6,1% | Menor que 9,0% |
| Parte não reconstruída de 16–20 Hz | 49,8% | 37% | Menor que 49,8% |
| Erro relativo total | 6,3% | 4,4% | Menor que 6,3% |
| Mediana do melhor `val_loss` dos 15 folds (`run.log`) | 0,0907 | 0,0726 | Menor que 0,0907 |
| η² da velocidade e do peso (`run.log`) | — | 0,10; 0,24 | Sem previsão de direção |

### 4.2 Avaliação (só reporte)

Sem previsão firme de direção.
- Se a energia criada fora da banda estiver mascarando o dano, a AUC das severidades 0,2 e 0,5 pode subir um pouco.
- Se a reconstrução melhorar igualmente para saudáveis e danificados, como no E002, a AUC fica igual.
- Severidades até 0,1 devem continuar com AUC ~0,5: nada no diagnóstico do E003 indica que o problema delas seja a arquitetura.

FPR e recall continuam afetados pela regra μ + 1σ, cuja fração de alarmes depende do formato da distribuição (ver E003, seção 9). A comparação deve se apoiar na AUC, que não depende do limiar, lembrando que ela também fica confundida com as épocas (seção 3).

### 4.3 Critério de adoção

Adotar a arquitetura se as duas condições valerem, contra o E003 com 100 épocas:
1. A fração do resíduo acima de 20 Hz no fold 1 cair em relação a 6,7%.
2. A mediana do melhor `val_loss` dos 15 folds não piorar em relação a 0,0907.

Se adotada, a nova base usa a arquitetura do E004 com 200 épocas (adotadas no E002), salvo decisão em contrário. O tempo de execução não entra no critério, mas será registrado.

## 5. Configuração e origem de cada escolha

Igual ao E003, exceto as linhas marcadas.

| Parâmetro | Valor | Origem | Justificativa / evidência |
|---|---|---|---|
| `nodes_to_use` | 8 nós (10654, 10659, 10664, 10669, 10658, 10663, 10668, 10673) | Herdado | Mesmos nós dos configs de OCSVM e Isolation Forest |
| `axes_to_use` | X, Y | Herdado | Mesmos eixos dos outros pipelines |
| `filter_type`, `cutoff_low`, `cutoff_high` | Passa-banda 1–20 Hz (Butterworth ordem 4, `filtfilt`) | Medido / Físico | Ver E003 |
| Ordem filtro/recorte | Filtra o registro completo, depois recorta | Físico | Ver E003 |
| Comprimento do sinal | Recorte em 1824 amostras (7,12 s) | Medido / Físico | 1824 é múltiplo de 8, então o recorte não muda com os novos pools |
| Normalização | Padronização por canal, só com saudáveis do treino | Físico / premissa | Ver E003 |
| Seleção de features | Nenhuma | Físico | Colunas são instantes de tempo |
| ★ `pool_sizes` | `[2, 2, 2]` | Medido / Físico | Latente a 32 amostras/s, mais perto da banda de 20 Hz. Ampliação em passos de ×2. Motivado pela energia criada fora da banda no E003 (seção 1) |
| ★ `encoder_filters` | `[32, 32, 4]` | Arbitrário / controle | 32 nas duas primeiras camadas: mantido do E003 (arbitrário). 4 canais no latente: escolhido só para manter o latente em 912 valores |
| ★ `kernel_size` | 19 | Físico (heurística) | Menor kernel que mantém o campo receptivo do encoder ≥ 2 períodos do modo de ~4 Hz (128 amostras). O fator "2 períodos" é uma escolha. Também cobre ≥ 1 período da maior frequência da banda (20 Hz → 13 amostras) |
| `l1_activity` | 1e-4 | Arbitrário | Ponto de partida do E001. Medido no E003: a penalidade equivale a 10% do MSE |
| Ativações e camadas | ReLU; saída linear; `UpSampling1D` + Conv | Convenção | Ver E001. O `UpSampling1D` é mantido de propósito: trocá-lo seria mais uma mudança |
| `learning_rate` | 1e-3 (Adam) | Convenção | Padrão do Adam |
| ★ `epochs` | 100 | Pedido | Custo por época ~2,8× maior; avaliação mais rápida. Comparação feita contra o E003 com 100 épocas (seção 3) |
| `early_stopping_patience` | 10 (com `restore_best_weights`) | Convenção | Valor comum |
| `batch_size` | 32 | Convenção | Valor comum |
| `validation_split` | 0,1, após embaralhar os saudáveis | Convenção | Ver E001 |
| Score de anomalia | MSE médio sobre 1824 × 16 | Convenção | Erro de reconstrução usual |
| Regra do limiar | μ + 1σ do erro dos saudáveis de treino | Pedido | Mantida para não misturar mudanças; revisão pendente (E003, seção 10) |
| `random_state` | 42 | Convenção | Semente fixa |

## 6. Verificação da premissa não supervisionada

- [x] Modelo, normalização, filtros e limiar ajustados só com saudáveis do treino.
- [x] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC. Os novos valores vêm do espectro dos saudáveis, de regras sobre a banda do sinal e de um pedido de tempo de execução.
- [x] Motivação e critério de adoção usam só saudáveis ou raciocínio físico.

## 7. Execução

Comandos (depois de commitar o código e esta pasta):

```powershell
.venv\Scripts\python.exe src\test_autoencoders.py outputs\autoencoders\E004_pooling-2x2x2 *> outputs\autoencoders\E004_pooling-2x2x2\run.log
.venv\Scripts\python.exe src\residual_spectrum.py outputs\autoencoders\E004_pooling-2x2x2
```

Duração estimada: 1,5–2,5 h para o run principal. O E003 levou ~2,1 s por época; com custo ~2,8× maior, são ~6 s por época, e no máximo 1500 épocas (15 folds × 100). Não medido. O diagnóstico do resíduo leva cerca de 15 min.

## 8. Resultados

## 9. Análise

## 10. Decisão e próximos passos

## Notas posteriores
