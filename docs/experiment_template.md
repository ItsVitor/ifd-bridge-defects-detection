# E{NNN}: {Título}

| Campo | Valor |
|---|---|
| Status | Proposto / Em execução / Concluído / Abandonado |
| Pai (base de comparação) | E{NNN} |
| Proposto em | AAAA-MM-DD |
| Executado em | AAAA-MM-DD |
| Código | commit `{hash}` |

## 1. Motivação

O que foi observado, no experimento pai ou numa análise à parte, que leva a este experimento. Cite os números e de onde vêm (pasta, seção do log, análise). Separe o que foi medido do que é suposição.

## 2. Hipótese

Uma frase que possa ser refutada pelo resultado:

> Se {mudança}, então {efeito esperado}, porque {mecanismo}.

## 3. Mudança em relação ao pai

| Item | Pai | Este experimento |
|---|---|---|
| | | |

Mudanças de código: arquivo e o que mudou.

Se houver mais de uma mudança: por que não foi possível separá-las e quais efeitos ficam confundidos.

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

O que se espera em `val_loss`, erro dos saudáveis de treino, η² por EOV, fração dos saudáveis de treino acima do limiar etc.

### 4.2 Avaliação (só reporte)

Direção esperada de AUC, recall e FPR por severidade. Não decide a adoção.

### 4.3 Critério de adoção

Condição, baseada só em 4.1 ou em raciocínio físico, para a mudança entrar na base.

## 5. Configuração e origem de cada escolha

Copie a tabela do pai e marque com ★ as linhas alteradas. Origens possíveis: Medido, Físico, Convenção, Herdado, Pedido, Arbitrário (ver `docs/experiments.md`).

| Parâmetro | Valor | Origem | Justificativa / evidência |
|---|---|---|---|
| `nodes_to_use` | | | |
| `axes_to_use` | | | |
| `filter_type`, `cutoff_low`, `cutoff_high` | | | |
| Comprimento do sinal (recorte) | | | |
| Normalização | | | |
| `encoder_filters` | | | |
| `pool_sizes` | | | |
| `kernel_size` | | | |
| `l1_activity` | | | |
| Ativações e camadas de saída | | | |
| `learning_rate` | | | |
| `epochs`, `early_stopping_patience` | | | |
| `batch_size` | | | |
| `validation_split` | | | |
| Score de anomalia | | | |
| Regra do limiar (`threshold_k`) | | | |

## 6. Verificação da premissa não supervisionada

- [ ] Modelo, normalização, filtros e limiar ajustados só com saudáveis do treino.
- [ ] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC.
- [ ] Motivação e critério de adoção usam só saudáveis ou raciocínio físico.

## 7. Execução

Comando, duração e observações: épocas em que o early stopping parou, avisos, falhas.

## 8. Resultados

### 8.1 Critérios só com saudáveis

### 8.2 Métricas globais

| Métrica | Pai | Este | Δ |
|---|---|---|---|
| Acurácia | | | |
| Recall | | | |
| FPR | | | |
| AUC agregada | | | |

### 8.3 Por severidade

Tabela "PERFORMANCE BY DAMAGE SEVERITY" do `run.log`.

### 8.4 Distribuição dos erros

![Distribuição do erro por severidade](error_distribution.png)

## 9. Análise

- Cada previsão da seção 4 contra o resultado: confirmada, refutada ou inconclusiva.
- Explicação do resultado, separando o que foi verificado do que é hipótese.
- Surpresas.

## 10. Decisão e próximos passos

- **Decisão:** adotada como nova base, descartada ou inconclusiva, aplicando o critério 4.3.
- **Próximos experimentos:** cada um com a observação que o motiva.

## Notas posteriores

Correções com data. As seções 1–6 não são reescritas depois da execução.
