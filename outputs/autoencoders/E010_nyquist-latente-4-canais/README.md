# E010: Latente a 64 amostras/s com 4 canais

| Campo | Valor |
|---|---|
| Status | Proposto |
| Pai (base de comparação) | [E008](../E008_penalidade-l1-1e-5/README.md) |
| Proposto em | 2026-11-08 |
| Executado em | — |
| Código | — (commitar antes do run) |

## 1. Motivação

**A pergunta do E009 continua aberta.** O E009 testou o latente a 64 amostras/s (Nyquist de 32 Hz, acima da banda de 20 Hz do filtro) e foi rejeitado: o MSE dos saudáveis de treino subiu de 0,0067 para 0,0076. Mas ele mudou duas coisas juntas, a resolução no tempo e os canais do latente (4 → 2), e não dá para separar as duas.

**O que o E009 mediu aponta para os 2 canais** (todas as medições só com saudáveis de treino):
- No fold 1, 1 dos 2 canais morreu. O latente útil caiu para 456 valores.
- O canal que sobrou quase nunca chegou a zero (3,2% de zeros). Isso é compatível com a ReLU: com um canal só, a rede precisa de um deslocamento constante para representar uma oscilação em torno de zero.
- No E008, com 4 canais e o mesmo coeficiente L1, nenhum canal morreu no fold 1.

**A faixa de 16–20 Hz continua a pior.** Em todos os experimentos desde o E004, ela é 6–10 vezes pior reconstruída que as outras faixas (7,6% de perda no fold 1 do E008, contra < 1% nas outras). Ela está acima do Nyquist do latente atual (16 Hz).

**Literatura.** [Manakov et al. (2019)](https://arxiv.org/abs/1911.07460) estudaram autoencoders convolucionais de imagens e concluíram que o tamanho espacial do latente pesa mais na generalização que o número de canais. Aqui, o tamanho espacial corresponde à duração do latente. O estudo não trata de detecção de anomalias e serve só como indicação.

**Primeiro experimento com o diagnóstico em todos os folds.** A partir do commit `fbd41a9`, o run principal mede o espectro do resíduo e o latente nos saudáveis de treino de cada um dos 15 folds. Os experimentos anteriores só têm o fold 1, que foi atípico no E008 (13º de 15 em MSE) e no E009 (1º de 15).

## 2. Hipótese

> Se a redução no tempo cair de 8× para 4× (pools `[2, 2, 1]`) mantendo os 4 canais do latente, então o erro dos saudáveis cai e a parte não reconstruída de 16–20 Hz diminui, porque o latente passa a 64 amostras/s (Nyquist de 32 Hz) sem o gargalo de 2 canais ReLU que derrubou o E009.

## 3. Mudança em relação ao pai

| Item | E008 | E010 |
|---|---|---|
| `pool_sizes` | `[2, 2, 2]` (redução de 8× no tempo; latente a 32 amostras/s) | `[2, 2, 1]` (redução de 4×; latente a 64 amostras/s) |

Sem mudanças de código no modelo. `encoder_filters` continua `[32, 32, 4]`.

**Efeito confundido: o latente dobra.** Ele passa de 228 × 4 = 912 para 456 × 4 = 1824 valores, e a compressão total cai de 32× (29.184 ÷ 912) para 16×. Não dá para manter 912 valores com 4 canais e 456 instantes. A alternativa de manter 912 valores foi o E009, com 2 canais.

| | Instantes no latente | Canais | Total | Valores por segundo | Compressão |
|---|---|---|---|---|---|
| E008 (pai) | 228 | 4 | 912 | 128 | 32× |
| E009 | 456 | 2 | 912 | 128 | 32× |
| E010 | 456 | 4 | 1824 | 256 | 16× |

**Como ler o resultado:**
- **E010 contra E008 (pai):** muda a resolução no tempo e, junto, o tamanho do latente.
- **E010 contra E009:** só os canais do latente mudam (2 → 4), com a mesma resolução no tempo. Essa comparação mostra quanto do resultado do E009 vinha dos 2 canais.

**Risco de reconstruir também o dano.** Um latente maior pode deixar o modelo mais perto de copiar a entrada e reconstruir bem também as passagens danificadas. Duas evidências contra esse risco: no E007, mais capacidade (sem L1) não piorou a detecção; e Manakov et al. não observaram cópia mesmo com latentes do tamanho da entrada. O critério de adoção só com saudáveis não consegue antecipar esse efeito.

**Parâmetros e custo.** O modelo tem os mesmos 63.688 parâmetros do E008 (contados com `build_model`), porque pooling e upsampling não têm parâmetros. A única convolução que passa a rodar com o dobro de instantes é a primeira do decoder, que é pequena (4 → 4 canais). O custo por época deve ficar parecido com o do E008. O campo receptivo do encoder vai de 134 para 130 amostras, ainda acima das 128 (2 períodos do modo de ~4 Hz).

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

As referências do E008 e do E009 para espectro e latente são só do fold 1. No E010, essas medidas passam a ser a mediana dos 15 folds (`residual_spectrum.txt`); a comparação com os pais fica limitada por isso.

| Critério | E008 | E009 | Previsão para o E010 |
|---|---|---|---|
| Mediana, entre folds, do MSE médio dos saudáveis de treino | 0,0067 | 0,0076 | Menor que 0,0067 |
| Parte não reconstruída de 16–20 Hz | 7,6% (fold 1) | 5,3% (fold 1) | Mediana menor que 5% |
| Energia criada fora da banda, relativa à entrada | 0,042% (fold 1) | 0,033% (fold 1) | Mediana menor que 0,042% |
| Canais ativos no latente | 4 de 4 (fold 1) | 1 de 2 (fold 1) | 4 de 4 em todos os folds |
| Zeros no latente | 45,0% (fold 1) | 51,6% (fold 1) | Sem previsão |
| Penalidade L1 ÷ MSE | 0,23 (fold 1) | 0,19 (fold 1) | Sem previsão |
| Erro do peso 0,90 ÷ outros pesos (mediana entre folds) | 1,34 | 1,43 | ≈ 1,3–1,4 |
| Folds parados pelo early stopping | 15 de 15 | 11 de 15 | Maioria |

### 4.2 Avaliação (só reporte)

Sem previsão de direção para a AUC das severidades 0,2 e 0,5. Há dois cenários:
- **O latente mais fino ajuda:** a reconstrução dos saudáveis melhora mais que a dos danificados, como no E004, e a AUC sobe.
- **O latente maior reconstrói também o dano:** a diferença entre saudáveis e danificados diminui e a AUC cai.

Os falsos negativos devem ficar perto dos 884 do E008, porque o limiar continua decidido pelo peso 0,90. Severidades até 0,1 devem continuar com AUC ≈ 0,5; a 0,01 está no limite do dano < 1% do `slightly_damaged`, que forma 93% dos saudáveis de treino.

### 4.3 Critério de adoção

Adotar `pool_sizes = [2, 2, 1]` na base se as duas condições valerem:
1. A mediana, entre folds, do MSE médio dos saudáveis de treino for menor que a do E008 (0,0067).
2. A mediana, entre folds, de canais ativos no latente for 4 de 4.

A parte não reconstruída de 16–20 Hz e a energia fora da banda testam a hipótese, mas não entram no critério, porque as referências dos pais são só do fold 1.

**Limite conhecido do critério.** Menor MSE dos saudáveis não garante melhor detecção (E002, E006, E007 e E008 mostraram isso), e aqui o latente dobra, o que pode facilitar a reconstrução do dano. A métrica de teste é só reportada e não decide.

## 5. Configuração e origem de cada escolha

Igual ao E008, exceto a linha marcada.

| Parâmetro | Valor | Origem | Justificativa / evidência |
|---|---|---|---|
| `nodes_to_use` | 8 nós (10654, 10659, 10664, 10669, 10658, 10663, 10668, 10673) | Herdado / Pedido | Mesmos nós dos pipelines OCSVM e Isolation Forest, para comparação direta |
| `axes_to_use` | X, Y | Herdado / Pedido | Idem |
| `filter_type`, `cutoff_low`, `cutoff_high` | Passa-banda 1–20 Hz (Butterworth ordem 4, `filtfilt`) | Medido / Físico | Ver E003 |
| Ordem filtro/recorte | Filtra o registro completo, depois recorta | Físico | Ver E003 |
| Comprimento do sinal | Recorte em 1824 amostras (7,12 s) | Medido / Físico | Ver E001; 1824 também é múltiplo de 4 |
| Normalização | Padronização por canal, só com saudáveis do treino | Físico / premissa | Ver E003 |
| Seleção de features | Nenhuma | Físico | Colunas são instantes de tempo |
| ★ `pool_sizes` | `[2, 2, 1]` | Físico / Medido | Nyquist do latente de 32 Hz, acima da banda de 20 Hz; 16–20 Hz é a pior faixa desde o E004 (seção 1) |
| `encoder_filters` | `[32, 32, 4]` | Arbitrário / controle | Adotado no E004; os 4 canais são mantidos para isolar o problema dos 2 canais do E009 |
| `kernel_size` | 19 | Físico (heurística) | Adotado no E004; campo receptivo de 130 amostras com os novos pools |
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
- [x] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC. A motivação vem do espectro do resíduo e do latente dos saudáveis (E008, E009) e do Nyquist do latente.
- [x] Motivação e critério de adoção usam só saudáveis ou raciocínio físico.

## 7. Execução

Comando (depois de commitar esta pasta). O espectro do resíduo de todos os folds sai no mesmo run (`residual_spectrum.txt` e `residual_spectrum.parquet`):

```powershell
.venv\Scripts\python.exe src\test_autoencoders.py outputs\autoencoders\E010_nyquist-latente-4-canais *> outputs\autoencoders\E010_nyquist-latente-4-canais\run.log
```

Antes de rodar em outra máquina: `uv sync`.

Duração estimada: o teto é de 3000 épocas (15 folds × 200). O E008 somou 1939 épocas e o E009, 2398 (~3h45). O custo por época deve ser parecido com o do E008 (seção 3); o diagnóstico em todos os folds acrescenta poucos segundos por fold.

## 8. Resultados

## 9. Análise

## 10. Decisão e próximos passos

## Notas posteriores
