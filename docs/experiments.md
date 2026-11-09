# Fluxo de Experimentos

Cada experimento tem os seus porquês escritos **antes** de rodar: o que motivou a mudança, o que se espera que aconteça e por quê. Depois de rodar, o resultado é comparado com essas previsões. A sequência de experimentos forma um caderno de laboratório que explica como cada escolha do modelo chegou ao valor atual.

## Estrutura

```
outputs/autoencoders/
├── README.md                    # índice: sequência de experimentos e decisões
└── E001_baseline-tempo/
    ├── README.md                # registro do experimento (modelo: docs/experiment_template.md)
    ├── config.yaml              # cópia congelada da configuração usada
    ├── run.log                  # saída completa do run
    ├── test_errors.parquet      # erro por amostra de teste, com metadados EOV
    ├── train_errors.parquet     # erro por amostra saudável de treino, por fold
    ├── residual_spectrum.txt    # espectro do resíduo e latente dos saudáveis de treino, por fold (resumo)
    ├── residual_spectrum.parquet # idem, todas as medidas por faixa
    └── error_distribution.png   # distribuição do erro por severidade
```

- **Nome da pasta:** `E{NNN}_{slug}`. O número é sequencial e nunca reutilizado. O slug é curto, em minúsculas e com hífens.
- **`config/autoencoders.yaml`** guarda a configuração da base atual. Um experimento novo começa copiando esse arquivo para a própria pasta e alterando só o que for testado.

## Ciclo de vida

| Status | O que existe na pasta | Próximo passo |
|---|---|---|
| Proposto | `README.md` (seções 1–6) e `config.yaml` | Revisão conjunta antes de rodar |
| Em execução | Código commitado; run em andamento | Aguardar o fim do run |
| Concluído | Seções 7–10 preenchidas; índice atualizado | Propor o próximo experimento |
| Abandonado | Motivo registrado no README | Nenhum; a pasta é mantida |

## Regras

1. **O porquê vem antes do resultado.** Motivação, hipótese, previsões e critério de adoção são escritos e revisados antes do run. Depois dele, não são reescritos: correções entram em "Notas posteriores", com data.
2. **Uma mudança por experimento**, em relação ao experimento pai. Se duas mudanças precisarem ir juntas, o README explica por quê e registra quais efeitos não podem ser separados.
3. **Toda escolha tem origem declarada.** A tabela de configuração classifica cada valor (ver abaixo). Escolhas sem justificativa própria são marcadas como "Arbitrário", não ganham uma justificativa inventada.
4. **Premissa não supervisionada** (`.kiro/steering/unsupervised-premise.md`). A motivação e o critério de adoção usam só dados saudáveis ou raciocínio físico. As métricas de teste são reportadas, mas não decidem a adoção.
5. **Nada é sobrescrito.** Rodar de novo, mesmo sem mudanças, gera um novo ID. Experimentos que deram errado continuam registrados.
6. **Reprodutibilidade.** A configuração fica congelada na pasta. O código é identificado pelo commit, que o script grava no log, avisando se há alterações não commitadas em `src/`.
7. **O índice é atualizado** ao fim de cada experimento, incluindo o número total de variantes já avaliadas nos folds de teste. Quanto mais variantes avaliadas nos mesmos folds, maior o risco de um resultado otimista por acaso.

## Origem das escolhas

| Origem | Significado | Exemplo |
|---|---|---|
| Medido | Derivado de uma medição nos dados saudáveis, citada no README | Passa-banda 1–20 Hz, a partir da PSD dos saudáveis |
| Físico | Derivado da física do problema ou da literatura de monitoramento estrutural | Kernel cobrindo um período da maior frequência modal |
| Convenção | Valor padrão comum em bibliotecas ou na literatura | Adam com `lr = 1e-3` |
| Herdado | Mantido de outra pipeline ou de um experimento anterior, sem justificativa própria | Nós e eixos dos pipelines OCSVM/IF |
| Pedido | Decisão explícita do autor do projeto | Limiar inicial μ + 1σ |
| Arbitrário | Ponto de partida sem justificativa; candidato a investigação | Filtros `[32, 32, 16]` |

## Como rodar

```powershell
New-Item -ItemType Directory outputs\autoencoders\E004_slug
Copy-Item config\autoencoders.yaml outputs\autoencoders\E004_slug\config.yaml
# editar config.yaml e README.md; revisar; commitar o código
.venv\Scripts\python.exe src\test_autoencoders.py outputs\autoencoders\E004_slug *> outputs\autoencoders\E004_slug\run.log
```
