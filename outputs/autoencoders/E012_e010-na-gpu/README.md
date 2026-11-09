# E012: Configuração do E010 na GPU

| Campo | Valor |
|---|---|
| Status | Proposto |
| Pai (base de comparação) | [E010](../E010_nyquist-latente-4-canais/README.md) |
| Proposto em | 2026-11-09 |
| Executado em | — |
| Código | — (commitar antes do run) |

## 1. Motivação

**Rodar na GPU deixa os experimentos ~3× mais rápidos.** O TensorFlow 2.20 só usa GPU dentro do WSL2. Medição no notebook do autor (GTX 1650), com dados sintéticos no formato real (1260 × 1824 × 16), mediana das épocas 2–6:

| Modelo | CPU (notebook) | GPU, com determinismo |
|---|---|---|
| `[32, 32, 4]`, pools `[2, 2, 1]` (E010) | 5,6 s/época | 1,8 s/época |
| `[32, 16, 4]`, pools `[2, 2, 1]` (E011) | 4,6 s/época | 1,6 s/época |

Na outra máquina do autor, que não tem GPU NVIDIA, o E010 levou 5,2 s/época (~3 h de treino para 2112 épocas).

**A troca de dispositivo muda os números.** A GPU faz as contas em outra ordem que a CPU, e a diferença de arredondamento cresce ao longo do treino; o early stopping pode parar em outra época. Comparar um experimento rodado na GPU com um pai rodado na CPU mistura a troca de dispositivo com a mudança testada. O E011 seria o primeiro caso: o critério de adoção dele ("MSE não maior que o do E010") é sensível a diferenças pequenas.

**Ainda não há estimativa da variação entre treinos.** Todos os experimentos usaram uma única semente, então não se sabe quanto o resultado muda sem mudar nada do modelo. A diferença entre CPU e GPU com a mesma configuração dá uma primeira estimativa dessa variação.

## 2. Hipótese

> Se a configuração do E010 for treinada na GPU (com determinismo das operações ligado), então as medidas só com saudáveis ficam próximas das do E010 (mediana do MSE dos saudáveis de treino a até 10% de 0,0033), porque o modelo, os dados e a semente são os mesmos e só a aritmética muda.

## 3. Mudança em relação ao pai

| Item | E010 | E012 |
|---|---|---|
| Dispositivo | CPU (outra máquina do autor, Windows) | GPU NVIDIA GTX 1650 (notebook do autor, WSL2 Ubuntu 24.04) |
| Determinismo | Padrão da CPU (já determinística com semente fixa) | `tf.config.experimental.enable_op_determinism()`, ligado só quando há GPU |

O `config.yaml` é uma cópia exata do E010 (mesmo hash). O código ganhou duas linhas: o determinismo só na GPU (em `train_model`) e a linha `Device:` no `run.log`. Na CPU, o caminho do código não muda.

**Ambiente.** Mesmas versões travadas do `uv.lock` (TensorFlow 2.20.0, Python 3.13), mais as bibliotecas CUDA de `tensorflow[and-cuda]==2.20.0`, num ambiente separado no Ubuntu (`~/venvs/ifd-bridge-gpu`). Passos em `docs/experiments.md`, seção "Na GPU (WSL2)".

**Confundido: máquina e sistema operacional.** Além do dispositivo, mudam a máquina (notebook, e não a outra do autor) e o sistema (Linux no WSL2, e não Windows). Não dá para separar esses efeitos; todos são "trocar de ambiente de execução".

## 4. Previsões (escritas antes de rodar)

### 4.1 Critérios só com saudáveis

| Critério | E010 (CPU) | Previsão para o E012 (GPU) |
|---|---|---|
| Mediana, entre folds, do MSE médio dos saudáveis de treino | 0,0033 (0,0028–0,0047) | Entre 0,0030 e 0,0036 (±10%) |
| Diferença relativa do MSE por fold (E012 ÷ E010 − 1), mediana dos módulos | — | Até 10% |
| Canais ativos no latente | 4 de 4 em 13 folds; 3 de 4 nos folds 1 e 4 | Mediana 4 de 4; os folds com canal morto podem mudar |
| Parte não reconstruída de 16–20 Hz | 3,1% (2,7–4,4%) | Entre 2,8% e 3,4% |
| Erro do peso 0,90 ÷ outros pesos (mediana entre folds) | 1,27 (1,23–1,41) | ≈ 1,27 |
| Épocas no total | 2112 | Diferente por fold; total na mesma ordem |
| Duração do treino | ~3 h (5,2 s/época, outra máquina) | ~1h10 (1,8 s/época × ~2100 épocas, mais ~3–4 min de carregamento) |

### 4.2 Avaliação (só reporte)

Falsos negativos, FPR e AUC por severidade perto dos do E010 (FN 891, FPR 6,7%, AUC sev. 0,2 0,75). A diferença entre os dois runs nessas métricas também é registrada como variação de execução, e não como efeito de uma mudança no modelo.

### 4.3 Critério de decisão

Não há adoção: a configuração é a mesma da base. O E012 decide como comparar os próximos experimentos:
1. **Se a mediana do MSE dos saudáveis ficar a até 10% do E010:** os próximos experimentos rodam na GPU e usam o E012 como referência. A diferença medida entre E010 e E012 passa a ser a margem mínima para considerar uma mudança de MSE como efeito, e não variação de execução.
2. **Se ficar a mais de 10%:** a variação de execução é grande demais para os critérios usados até aqui (por exemplo, o "não maior" do E011). Antes de seguir, é preciso discutir com o autor como estimar essa variação (por exemplo, mais de uma semente).

Os 10% são **Arbitrário**: não há medição anterior da variação entre execuções.

## 5. Configuração e origem de cada escolha

Igual ao E010 (cópia exata do `config.yaml`); ver a seção 5 do [E010](../E010_nyquist-latente-4-canais/README.md). A única diferença é o ambiente de execução:

| Item | Valor | Origem | Justificativa / evidência |
|---|---|---|---|
| Dispositivo | GPU (GTX 1650) no WSL2 | Pedido / Medido | Pedido do autor (executar mais rápido); ~3× mais rápido com determinismo (seção 1) |
| Determinismo na GPU | Ligado | Pedido / Físico | Pedido do autor; sem ele, o cuDNN dá resultados diferentes a cada execução e a semente não basta |
| Margem de 10% (seção 4.3) | 10% | Arbitrário | Sem medição anterior da variação entre execuções |

## 6. Verificação da premissa não supervisionada

- [x] Modelo, normalização, filtros e limiar ajustados só com saudáveis do treino (configuração idêntica à do E010).
- [x] Nenhum parâmetro escolhido com danificados, métricas de teste ou AUC.
- [x] O critério de decisão usa só o MSE dos saudáveis.

## 7. Execução

Commitar esta pasta e o código antes. No terminal do Ubuntu (WSL2):

```bash
cd /mnt/c/Users/vitor/OneDrive/Documentos/GitHub/ifd-bridge-defects-detection
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.autocrlf GIT_CONFIG_VALUE_0=true \
  ~/venvs/ifd-bridge-gpu/bin/python src/test_autoencoders.py outputs/autoencoders/E012_e010-na-gpu \
  > outputs/autoencoders/E012_e010-na-gpu/run.log 2>&1
```

A linha `Device:` do `run.log` deve dizer `GPU [...], op determinism on`. O teto é de 3000 épocas (15 folds × 200); o E010 somou 2112.

Verificações feitas antes do run (2026-11-09, no notebook):
- O ambiente do Ubuntu enxerga a GPU (`tf.config.list_physical_devices("GPU")`).
- Dois treinos seguidos com determinismo ligado (dados sintéticos, 3 épocas) deram erros idênticos.
- O carregamento completo dos dados no WSL levou ~3,5 min e chegou a 14,7 GB de pico; o limite do WSL foi aumentado para 24 GB.

## 8. Resultados

## 9. Análise

## 10. Decisão e próximos passos

## Notas posteriores
