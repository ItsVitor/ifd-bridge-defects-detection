# E017: Limiar por velocidade (entrada log-PSD, sem retreino)

| Campo | Valor |
|---|---|
| Status | Proposto |
| Pai (base de comparação) | [E015](../E015_log-psd-500-epocas/README.md) |
| Proposto em | 2026-11-12 |
| Executado em | — |
| Código | — (commitar antes do run; mesmo código do E016) |

## 1. Motivação

**Na entrada log-PSD, a dependência do erro em relação à velocidade é bem maior que no tempo.** Medido só com os saudáveis de treino do E015 (`train_errors.parquet`):
- A velocidade explica 58% da variação do erro dos saudáveis (η² 0,578; no E012, 0,043).
- O erro mediano a 45 km/h é ~3× o de 55 km/h.
- Com um limiar único por fold, as passagens acima dele são, em média por fold, 48,1 a 45 km/h, 10,1 a 50 km/h e 4,7 a 55 km/h. Na irregularidade 1, a fração acima do limiar é 10,3%, 1,3% e 0,0%.

A 55 km/h, o limiar único fica acima de todos os saudáveis de irregularidade 1 daquela velocidade. No teste do E015, o recall da severidade 0,2 a 55 km/h foi de 1% (só reporte).

**Por que testar também na PSD, se ela não foi adotada.** O E015 descartou a entrada log-PSD porque a distribuição do erro dos saudáveis ficou larga (razão p95 ÷ mediana 2,113). Boa parte dessa largura vem da diferença de nível entre velocidades, que é justamente o que o limiar por velocidade compensa. Este experimento mede o mesmo procedimento do E016 onde o efeito da velocidade é maior. É um diagnóstico, não um candidato à base (seção 4.3).

## 2. Hipótese

> Se cada fold usar um limiar por velocidade (o percentil 95 dos saudáveis de treino daquela velocidade), em vez de um único limiar, então a fração de saudáveis de treino acima do limiar passa a ser 5% em cada velocidade, porque o percentil é calculado separadamente sobre os saudáveis de cada uma.

A hipótese vale por construção, como no E005 e no E016.

## 3. Mudança em relação ao pai

| Item | E015 | E017 |
|---|---|---|
| Limiar de cada fold | 1 limiar: percentil 95 dos 1260 saudáveis de treino | 3 limiares: percentil 95 dos 420 saudáveis de treino de cada velocidade |
| Limiar aplicado a cada passagem de teste | O do fold | O da velocidade da passagem, no fold |

Nada mais muda: os erros de reconstrução são os do E015.

**Implementação:** a mesma do E016 (seção 3 do [E016](../E016_limiar-por-velocidade-tempo/README.md)): `src/rethreshold.py` com `threshold_group: "Velocidade"`, sobre os erros salvos do E015. Os limites também são os mesmos (velocidade contínua em operação; ~21 passagens acima de cada limiar).

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

Calculado antes do registro, só com os saudáveis de treino do E015:

| Critério | E015 | E017 (calculado) |
|---|---|---|
| Saudáveis de treino acima do limiar, por velocidade | 11,5% / 2,4% / 1,1% (48,1 / 10,1 / 4,7 de 420) | 5,0% em cada velocidade (21 de 420) |
| Irregularidade 1 acima do limiar, por velocidade | 10,3% / 1,3% / 0,0% | 3,7% / 4,0% / 3,8% |
| Limiar por velocidade ÷ limiar único (mediana entre folds) | 1 | 45 km/h 1,13; 50 km/h 0,89; 55 km/h 0,38 |
| Acima do limiar no treino, por fold (grupo das 15 / peso 0,90 / outros) | 14 / 27,0 / 22,0 | 14 / 36,4 / 12,6 |
| Grupo irregularidade 2 × módulo 0,975 acima do limiar | Todo | Todo (~4,7 por velocidade e fold) |

O limiar de 55 km/h cai para ~38% do limiar único, uma mudança muito maior que no E016 (85%).

### 4.2 Avaliação (só reporte)

- **AUC por severidade e AUC agregada do `run.log`: idênticas às do E015, por construção** (erro bruto). A AUC do erro ÷ limiar da velocidade será calculada na análise como diagnóstico.
- **FPR por velocidade:** cai a 45 km/h (26,2% no E015) e sobe a 55 km/h (1,3%), aproximando as três.
- **Recall a 55 km/h:** deve subir mais que no E016, porque o limiar dessa velocidade cai ~62%.
- **Falsos negativos:** sem previsão de direção no total; devem subir a 45 km/h e cair a 55 km/h.
- Severidades até 0,05 devem continuar com recall perto do FPR.

### 4.3 Critério de adoção

**Nenhum: este experimento não é candidato à base.** A entrada log-PSD foi descartada no E015 pelo critério só com saudáveis decidido antes daquele run. Escolher entre E016 e E017 pelos resultados de teste violaria a premissa não supervisionada. Os resultados servem para mostrar quanto da dependência da velocidade o limiar compensa em cada representação.

Para reabrir a PSD seria preciso um critério só com saudáveis, decidido antes, que leve em conta o limiar por velocidade. A medida natural já existe: a razão p95 ÷ mediana do erro dos saudáveis sem o nível de cada velocidade, que foi 1,681 no E015, contra 1,438 no E012. Com ela, a PSD continua mais larga que o tempo.

## 5. Configuração e origem de cada escolha

Igual ao E015, exceto a linha marcada. Ver a tabela completa no [E015](../E015_log-psd-500-epocas/README.md), seção 5.

| Parâmetro | Valor | Origem | Justificativa / evidência |
|---|---|---|---|
| Entrada, modelo e treino | Os do E015 (log-PSD 40 × 16, kernel 5, `[32, 32, 4]`, `[2, 2, 1]`, teto de 500 épocas) | Herdado | Os erros são reaproveitados |
| Regra do limiar | Percentil 95 dos saudáveis de treino | Pedido / Medido | Adotada no E005 (α = 5%) |
| ★ `threshold_group` | `Velocidade` | Pedido / Medido / Físico | Mesma justificativa do E016; aqui, com um limiar único, 55 km/h opera com α ≈ 0% na irregularidade 1 (seção 1) |

## 6. Verificação da premissa não supervisionada

- [x] Modelo, normalização, filtros e limiar ajustados só com saudáveis do treino.
- [x] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC.
- [x] Não há critério de adoção; os resultados de teste não serão usados para escolher entre E016 e E017.

## 7. Execução

Commitar o código e esta pasta antes. No PowerShell, em menos de um minuto:

```powershell
.venv\Scripts\python.exe src\rethreshold.py outputs\autoencoders\E015_log-psd-500-epocas outputs\autoencoders\E017_limiar-por-velocidade-psd *> outputs\autoencoders\E017_limiar-por-velocidade-psd\run.log
```

O `run.log` deve mostrar `threshold_group=Velocidade` e 45 linhas de limiar, cada uma com `healthy train above: 0.050`.

## 8. Resultados

## 9. Análise

## 10. Decisão e próximos passos

## Notas posteriores

