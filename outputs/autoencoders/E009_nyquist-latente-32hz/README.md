# E009: Latente a 64 amostras/s (Nyquist de 32 Hz)

| Campo | Valor |
|---|---|
| Status | Proposto |
| Pai (base de comparação) | [E008](../E008_penalidade-l1-1e-5/README.md) |
| Proposto em | 2026-11-08 |
| Executado em | — |
| Código | — (commitar antes do run) |

## 1. Motivação

**A faixa de 16–20 Hz é a pior reconstruída em todos os experimentos desde o E004.** Parte não reconstruída por faixa, fold 1, só saudáveis de treino (`residual_spectrum.txt` de cada experimento):

| Faixa (Hz) | 1–4 | 4–8 | 8–12 | 12–16 | 16–20 |
|---|---|---|---|---|---|
| E006 | 0,6% | 0,7% | 0,4% | 0,9% | 7,1% |
| E007 | 0,4% | 0,4% | 0,2% | 0,6% | 5,2% |
| E008 | 0,6% | 0,6% | 0,4% | 0,9% | 7,6% |

Nas outras faixas a perda fica abaixo de 1%; em 16–20 Hz, ela é 6–10 vezes maior.

**A faixa está acima do Nyquist do latente.** Um sinal amostrado a *f* amostras por segundo só representa sem ambiguidade frequências até *f*/2 (a frequência de Nyquist). Com pools `[2, 2, 2]`, o latente tem 1824 ÷ 8 = 228 instantes em 7,125 s, ou 32 por segundo: o Nyquist é 16 Hz. A faixa de 16–20 Hz só é reconstruída (~93% dela) porque o decoder combina vários canais do latente para recriá-la, e isso custa erro.

**O mesmo mecanismo já funcionou uma vez.** No E004, reduzir menos o tempo (32× → 8×) mantendo o latente em 912 valores reduziu o erro de todas as faixas de 1–20 Hz em 5–10× e a energia criada fora da banda em ~11×.

**O decoder ainda cria energia fora da banda.** No E008, 6,8% do resíduo do fold 1 está acima de 20 Hz, onde a entrada não tem energia (0,042% da energia de entrada). A explicação mais provável, não verificada, são os "degraus" do `UpSampling1D`. Com um passo de ×2 a menos, essa energia deve cair.

## 2. Hipótese

> Se a redução no tempo cair de 8× para 4× (pools `[2, 2, 1]`), mantendo o latente em 912 valores e o campo receptivo do encoder acima de 128 amostras, então a parte não reconstruída de 16–20 Hz e a energia criada acima de 20 Hz diminuem, porque o latente passa a 64 amostras/s (Nyquist de 32 Hz, acima de toda a banda do filtro) e o decoder amplia o sinal em um passo de ×2 a menos.

## 3. Mudança em relação ao pai

| Item | E008 | E009 |
|---|---|---|
| `pool_sizes` | `[2, 2, 2]` (redução de 8× no tempo; latente a 32 amostras/s) | `[2, 2, 1]` (redução de 4×; latente a 64 amostras/s) |
| `encoder_filters` | `[32, 32, 4]` | `[32, 32, 2]` |

Sem mudanças de código. `MaxPooling1D(1)` e `UpSampling1D(1)` não alteram o sinal, então o último bloco do encoder fica só com a convolução. O recorte continua em 1824 amostras (1827 é cortado no múltiplo de 4 mais próximo, que também é 1824).

### Por que dois parâmetros mudam

A mudança testada é uma só, como no E004: onde a compressão acontece, no tempo ou nos canais. Os filtros da última camada do encoder caem de 4 para 2 para manter o tamanho do latente:

| | Instantes no latente | Canais do latente | Total | Valores por segundo |
|---|---|---|---|---|
| E008 | 228 | 4 | 912 | 32 × 4 = 128 |
| Só trocando os pools | 456 | 4 | 1824 | 256 |
| E009 | 456 | 2 | 912 | 64 × 2 = 128 |

**Por que manter 912 valores.** Não há um racional físico para 912: o valor vem dos pontos de partida arbitrários do E001 (57 instantes × 16 canais) e é mantido só como controle. Com o latente fixo, uma diferença no resultado pode ser atribuída à resolução no tempo, e não a um latente maior. Dobrar o latente também mudaria quanto o modelo comprime.

**Campo receptivo.** Com kernel 19, o campo receptivo do encoder vai de 134 para 130 amostras, ainda acima das 128 (2 períodos do modo de ~4 Hz). O kernel não precisa mudar.

**Parâmetros e custo.** O modelo passa de 63.688 para 61.024 parâmetros (contados com `build_model`). As convoluções de 32 filtros rodam nas mesmas resoluções de antes; só a primeira convolução do decoder, que é pequena (2 filtros), roda a 456 instantes em vez de 228. O custo por época deve ficar parecido com o do E008.

**Efeito confundido: 2 canais no latente.** Com 2 canais, cada canal precisa carregar mais da resposta. Há dois riscos:
- **ReLU:** o latente só tem valores positivos. Um sinal que oscila em torno de zero precisa de um deslocamento constante (que a L1 penaliza) ou de dois canais (um para a parte positiva, outro para a negativa). Com 4 canais isso é contornável; com 2, é mais apertado.
- **Canal morto:** se um canal morrer, o latente útil cai pela metade. No E008, com 1e-5, nenhum dos 4 canais morreu.

Há uma evidência de que 2 canais podem bastar: no E006, só 2 dos 4 canais estavam ativos no fold 1 (64 valores por segundo) e o modelo reconstruiu 99,4% da energia. Se o E009 piorar, não dá para separar o efeito da resolução no tempo do efeito de ter só 2 canais.

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

O critério de reconstrução é o MSE puro, de `train_errors.parquet`, e não o `val_loss` (que inclui a penalidade).

| Critério | E008 | Previsão para o E009 |
|---|---|---|
| Mediana, entre folds, do MSE médio dos saudáveis de treino | 0,0067 (0,0050–0,0079) | Menor |
| Parte não reconstruída de 16–20 Hz, fold 1 | 7,6% | Menor; deixa de ser 6–10× pior que as outras faixas |
| Energia criada acima de 20 Hz, relativa à entrada, fold 1 | 0,042% | Menor |
| Canais ativos no latente (fold 1) | 4 de 4 | 2 de 2 |
| Zeros no latente (fold 1) | 45,0% | Menor (cada canal precisa ficar ativo mais vezes) |
| Penalidade L1 ÷ MSE (fold 1) | 0,23 | Sem previsão (o E008 mostrou que essa razão não segue o coeficiente) |
| Erro do peso 0,90 ÷ outros pesos | 1,34 | ≈ 1,3 (quatro experimentos de reconstrução não a mudaram) |
| Folds parados pelo early stopping | 15 de 15 | Maioria |

### 4.2 Avaliação (só reporte)

Sem previsão de direção para a AUC das severidades 0,2 e 0,5. Melhorar 16–20 Hz só ajuda a detectar se o dano mudar essa faixa, e isso não se sabe nem deve ser verificado com os danificados antes da avaliação. Os falsos negativos devem ficar perto de 884, porque o limiar continua decidido pelo peso 0,90.

Severidades até 0,1 devem continuar com AUC ≈ 0,5. Para a severidade 0,01 há agora um motivo direto: o `slightly_damaged`, tratado como saudável, tem dano < 1% (informado pelo autor), e ele forma 93% dos saudáveis de treino. Um dano de 1% está no limite do que o modelo aprende como saudável.

### 4.3 Critério de adoção

Adotar `pool_sizes = [2, 2, 1]` e `encoder_filters = [32, 32, 2]` na base se as duas condições valerem:
1. A mediana, entre folds, do MSE médio dos saudáveis de treino for menor que a do E008 (0,0067).
2. O fold 1 tiver os 2 canais do latente ativos.

A parte não reconstruída de 16–20 Hz e a energia fora da banda testam a hipótese, mas não entram no critério, porque são medidas só no fold 1, que foi atípico no E008 (13º de 15 em MSE).

**Limite conhecido do critério.** Menor MSE dos saudáveis não garante melhor detecção (E002, E006, E007 e E008 mostraram isso). A métrica de teste é só reportada e não decide.

## 5. Configuração e origem de cada escolha

Igual ao E008, exceto as linhas marcadas.

| Parâmetro | Valor | Origem | Justificativa / evidência |
|---|---|---|---|
| `nodes_to_use` | 8 nós (10654, 10659, 10664, 10669, 10658, 10663, 10668, 10673) | Herdado / Pedido | Mesmos nós dos pipelines OCSVM e Isolation Forest, para comparação direta |
| `axes_to_use` | X, Y | Herdado / Pedido | Idem |
| `filter_type`, `cutoff_low`, `cutoff_high` | Passa-banda 1–20 Hz (Butterworth ordem 4, `filtfilt`) | Medido / Físico | Ver E003 |
| Ordem filtro/recorte | Filtra o registro completo, depois recorta | Físico | Ver E003 |
| Comprimento do sinal | Recorte em 1824 amostras (7,12 s) | Medido / Físico | Ver E001 |
| Normalização | Padronização por canal, só com saudáveis do treino | Físico / premissa | Ver E003 |
| Seleção de features | Nenhuma | Físico | Colunas são instantes de tempo |
| ★ `pool_sizes` | `[2, 2, 1]` | Físico / Medido | Nyquist do latente de 32 Hz, acima da banda de 20 Hz; 16–20 Hz é a pior faixa desde o E004 (seção 1) |
| ★ `encoder_filters` | `[32, 32, 2]` | Arbitrário / controle | Último valor ajustado para manter o latente em 912 valores (456 × 2); 912 vem do E001 e é Arbitrário. Os 32 filtros das outras camadas são Herdados do E001 |
| `kernel_size` | 19 | Físico (heurística) | Adotado no E004; campo receptivo de 130 amostras com os novos pools, acima de 128 |
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
- [x] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC. A motivação vem do espectro do resíduo dos saudáveis e do Nyquist do latente.
- [x] Motivação e critério de adoção usam só saudáveis ou raciocínio físico.

## 7. Execução

Comandos (depois de commitar esta pasta):

```powershell
.venv\Scripts\python.exe src\test_autoencoders.py outputs\autoencoders\E009_nyquist-latente-32hz *> outputs\autoencoders\E009_nyquist-latente-32hz\run.log
.venv\Scripts\python.exe src\residual_spectrum.py outputs\autoencoders\E009_nyquist-latente-32hz *> outputs\autoencoders\E009_nyquist-latente-32hz\residual_spectrum.log
```

Antes de rodar em outra máquina: `uv sync`.

Duração estimada: o teto é de 3000 épocas (15 folds × 200). O E008 somou 1939 épocas; o custo por época deve ser parecido (seção 3), mas o ponto de parada pode mudar. O diagnóstico do resíduo leva ~20–30 min.

**Limitação:** canais ativos, zeros e o espectro do resíduo são medidos só no fold 1. O critério 4.3 (2) depende dessa medição única.

## 8. Resultados

## 9. Análise

## 10. Decisão e próximos passos

## Notas posteriores
